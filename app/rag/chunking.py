"""
切分层：DocumentBlock -> 父块 + 子块。

这是整个 RAG 效果最关键的环节之一，也是简历上"父子块 + 语义切分"的落地点。

为什么要父子块（面试必考）：
    检索精度与上下文完整性是矛盾的：
      - 块切得小（如 300 字）：向量表示聚焦、召回准，但喂给大模型时上下文缺失，
        容易出现"答案对了但说不全/引用不完整"。
      - 块切得大（如 2000 字）：上下文完整，但向量被多主题稀释，召回不准。
    解法：两层结构。
      子块（小块）用于「召回」——命中率高；
      父块（大块）用于「生成」——命中子块后，把它的父块全文交给大模型。
    这样同时拿到"召回准"和"上下文全"，代价是多存一份文本（可忽略）。

两种切分策略（可配置，对应消融实验）：
    recursive —— 按分隔符递归切，快、不花钱，是默认策略。
    semantic  —— 用 embedding 计算相邻句相似度，在语义断层处断开，
                 更贴合语义边界，但要额外调用 embedding 接口（慢、有成本）。
    默认 recursive，把 semantic 作为可选项，是"效果 vs 成本"的显式权衡。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.core.logging import get_logger
from app.rag.parser import DocumentBlock

logger = get_logger(__name__)

# 段落级分隔符：给"父块聚合"用（父块要尽量保留完整段落）
_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", ". ", "; ", "，", ", ", " ", ""]

# 子块级分隔符：**句号优先**，且去掉单换行 "\n"
#
# 为什么句号要排在换行前面（实测踩到的坑）：
#   recursive_split 只在"当前块超过目标长度"时才会用下一级分隔符继续切。
#   像"第二章 年假标准"这种 250 字的章节，如果以 "\n" 为主分隔符，
#   它会被当成一个整体直接返回（因为 250 < 300），于是整章变成一个子块 ——
#   而这一章里其实讲了三件不同的事（年假天数 / 结转规则 / 请假流程），
#   揉在一个向量里会被互相稀释，检索精度下降。
#   把"。"提到最前面，就会先按句子拆开、再按目标长度重新聚合，
#   得到"几条语义相近的完整句子"组成的块，主题更单一。
#
# 为什么去掉单换行 "\n"：
#   中文制度类文档常是"一行一条"，行边界本身没有语义。
#   按 "\n" 切会让"单次连续休假超过5个工作日"这类条款被切成前后两半。
_CHILD_SEPARATORS = ["。", "；", "！", "？", "\n\n", ". ", "; ", "，", ", ", "\n", " ", ""]

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])")

# 中文条款编号前缀："第一条" / "第 3 条" / "（二）" / "一、"
# 语义切分时把它从句首剥掉：这类编号在所有条款里都出现，
# 会让不同条款的向量互相靠近、削弱"语义断层"信号。
_ARTICLE_PREFIX = re.compile(
    r"^\s*(?:第\s*[一二三四五六七八九十百零\d]+\s*[条章节款项]|"
    r"[一二三四五六七八九十]+\s*[、\.．]|[（(]\s*[一二三四五六七八九十\d]+\s*[)）])\s*"
)


@dataclass
class ChildChunk:
    """子块：用于向量召回的检索单元。"""

    chunk_uid: str
    content: str
    chunk_index: int
    parent_index: int          # 指向同一文档内的父块序号
    page_no: int | None
    section: str | None
    block_type: str
    token_count: int = 0


@dataclass
class ParentChunk:
    """父块：用于喂给大模型的上下文单元。"""

    parent_index: int
    content: str
    page_no: int | None
    section: str | None
    block_type: str


@dataclass
class ChunkResult:
    parents: list[ParentChunk]
    children: list[ChildChunk]

    @property
    def stats(self) -> dict:
        return {
            "parents": len(self.parents),
            "children": len(self.children),
            "avg_child_len": round(
                sum(len(c.content) for c in self.children) / max(len(self.children), 1), 1
            ),
        }


def new_chunk_uid(document_id: int, version: int, chunk_index: int) -> str:
    """
    确定性地生成 chunk_uid。

    为什么不用随机 uuid（面试可讲）：
        1. 幂等：同一篇文档重复上传/重建索引时 uid 不变，
           不会在向量库里留下两套重复数据；
        2. 可更新：文档更新时能按 uid 前缀精确定位并删除旧版本向量；
        3. 可复现：评测集里标注的 gold chunk_uid 在多次重建后依然有效，
           否则每次重建索引评测结果就没法对比了。
    """
    return f"d{document_id}v{version}c{chunk_index}"


# ============================================================
# 递归切分（默认策略）
# ============================================================
def recursive_split(text: str, size: int, overlap: int, separators: list[str] | None = None) -> list[str]:
    """
    按分隔符层级递归切分，带重叠。

    重叠（overlap）为什么必要：
        如果一个句子刚好被切断，前半句在块 A、后半句在块 B，
        那么用前半句提问时，块 B 检索不到、块 A 又答不全。
        overlap 让边界内容在两个块里各出现一次，缓解"跨块信息丢失"。

    递归为什么优于固定窗口：
        固定长度硬切会把一句话、一个表格切成两半；
        优先在"\\n\\n"（段落）处切，段落太长再退到句号，实在不行才硬切字符 ——
        尽可能保住语义边界。

    ⚠️ 重叠必须"整句对齐"（实测踩到的严重坑）：
        最初实现是直接取上一块末尾 N 个**字符**拼到下一块开头，
        结果切出了 "需报部门总监审批员工申请年假需提前3个工作日" 这种缝合文本 ——
        两句话各取一半接在一起，语义完全错乱，还会污染向量表示。
        正确做法是：重叠部分取"完整的最后一句"，保证拼接处的语义是干净的。
    """
    separators = separators or _SEPARATORS
    if len(text) <= size:
        return [text] if text.strip() else []

    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        # 兜底：按字符硬切（极少走到这里，中文有句号、英文有空格）
        chunks = []
        step = max(size - overlap, 1)
        for i in range(0, len(text), step):
            piece = text[i:i + size]
            if piece.strip():
                chunks.append(piece)
        return chunks

    parts = text.split(sep)
    pieces: list[str] = []
    buffer = ""
    for part in parts:
        candidate = (buffer + sep + part) if buffer else part
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer)
            # 单段仍超长 -> 用更细的分隔符继续递归
            if len(part) > size:
                pieces.extend(recursive_split(part, size, overlap, rest))
                buffer = ""
            else:
                buffer = part
    if buffer:
        pieces.append(buffer)

    # 整句对齐的重叠
    if overlap > 0 and len(pieces) > 1 and sep in ("。", "；", "！", "？", "\n", "\n\n"):
        overlapped = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            tail = prev[-(overlap + 20):]          # 多取一点，便于向前找到句首
            # 找到 tail 里最后一个句子边界之后的内容，作为完整的前置句
            cut = -1
            for mark in ("。", "；", "！", "？", "\n"):
                pos = tail.rfind(mark)
                cut = max(cut, pos)
            prefix = tail[cut + 1:].strip() if cut >= 0 else ""
            if prefix and len(prefix) <= overlap + 30:
                overlapped.append(f"{prefix}{sep}{cur}" if sep != "\n" else f"{prefix}\n{cur}")
            else:
                overlapped.append(cur)
        pieces = overlapped

    return [p.strip() for p in pieces if p.strip()]


# ============================================================
# 语义切分（可选策略）
# ============================================================
async def semantic_split(
    text: str, size: int, threshold: float, min_size: int = 0
) -> list[str]:
    """
    语义切分：计算相邻句子的向量相似度，在"语义断层"处断开。

    原理：
        同一个主题的相邻句，向量相似度高；话题切换时相似度骤降。
        把相似度低于阈值的位置作为切分点，得到的块"每块讲一件事"，
        向量表示更纯，召回更准。

    ⚠️ 为什么用**百分位**而不是绝对阈值（实测数据驱动的修正，值得讲）：
        最初的实现是 `相似度 < 0.62 就断开`。实测打印相似度分布后发现：
          制度类文档相邻句相似度几乎全落在 0.65~0.80，
          技术手册类因为术语重合多、整体更高。
        也就是说**相似度的绝对尺度随文本类型变化**，不存在跨文档通用的魔法阈值。
        固定 0.62 的后果：有的文档一个断点都没有（完全切不开），
        另一篇又误切在条款中间（切出"结婚证；║产假按国家规定执行"这种半截条款）。
        改成"取相似度**最低的若干处**作为断点"之后：
          - 断点数由块长目标(size)决定，而不是由猜出来的阈值决定；
          - threshold 从"绝对门槛"变成"操作点"（取最低的百分之多少），
            既可解释、又可调，且跨文档稳定。

    ⚠️ 还必须有「最小块长」约束（同样是实测踩坑）：
        最初没有任何合并机制，切出了 `超出标准部分需自行承担。`（12 字）、
        `第九条 出差交通标准：市内交通凭票据实报实销；`（23 字）这类碎片 ——
        条款后半句被切走，命中后喂给大模型的信息不完整，比切得粗更糟。
        现在改为：只有在当前块已达到 min_size 时才允许断开。

    代价：
        每个句子都要向量化（有 API 成本与延迟），实测切一篇 1500 字文档约 2 秒，
        所以只作为可配置策略、不作为默认（默认 recursive，0 额外成本）。
    """
    from app.rag.embedding import cosine_similarity, get_embedder

    # 把"第X条/第X章"这类编号从句首剥离，避免编号词主导向量表示
    # （编号在所有条款句里都出现，会把不同条款的向量拉近，削弱语义边界信号）
    def normalize(s: str) -> str:
        return _ARTICLE_PREFIX.sub("", s).strip() or s

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        return recursive_split(text, size, settings.CHUNK_OVERLAP)

    if min_size <= 0:
        min_size = max(int(size * 0.5), 80)       # 默认下限：目标长度的一半

    # 为了让相邻句的相似度更纯粹，在句首加一点章节上下文再向量化
    head = sentences[0][:24]
    probe_texts = [f"{head} {normalize(s)}" for s in sentences]

    embedder = await get_embedder()
    vectors = await embedder.embed_documents(probe_texts)

    sims = [cosine_similarity(vectors[i - 1], vectors[i]) for i in range(1, len(sentences))]

    # 目标断点数由"句子数 × 操作点比例"决定（threshold 越小 -> 断点越多）
    # 例：threshold=0.62 -> 取相似度最低的约 38% 位置作为断点
    target_breaks = max(1, int(len(sentences) * min(0.5, 1.0 - threshold)))
    order = sorted(range(len(sims)), key=lambda i: sims[i])
    break_positions = set(order[:target_breaks])

    groups: list[list[str]] = [[sentences[0]]]
    cur_len = len(sentences[0])
    for i in range(1, len(sentences)):
        too_long = cur_len + len(sentences[i]) > size
        semantic_break = (i - 1) in break_positions     # 该位置属于"相似度最低的一批"
        # 允许断开的前提：当前块已经攒够信息量（或已超长）
        can_break = cur_len >= min_size or too_long

        if (semantic_break or too_long) and can_break:
            groups.append([sentences[i]])
            cur_len = len(sentences[i])
        else:
            groups[-1].append(sentences[i])
            cur_len += len(sentences[i])

    chunks = [g for g in ("".join(g) for g in groups) if g.strip()]
    # 语义切分可能产出超长块（连续同主题），用递归兜底压到目标长度
    final: list[str] = []
    for c in chunks:
        if len(c) > size * 1.5:
            final.extend(recursive_split(c, size, settings.CHUNK_OVERLAP))
        else:
            final.append(c)
    return final


# ============================================================
# 对外主函数
# ============================================================
async def build_chunks(blocks: list[DocumentBlock], document_id: int = 0, version: int = 1) -> ChunkResult:
    """
    把解析出的块切成「父块 + 子块」两层。

    切分规则：
      1. 标题块不参与向量化，但会作为后续块的 section 上下文（标题本身没有检索价值）；
      2. 表格块整体作为一个单元（表格被切断就失去意义），超长时才递归切；
      3. 相邻的正文块先合并到"父块"目标长度，再在父块内部切子块 ——
         这样父块的上下文是完整的段落群，而不是零散单句。
    """
    strategy = settings.CHUNK_STRATEGY.lower()
    child_size = settings.CHUNK_SIZE
    parent_size = settings.PARENT_CHUNK_SIZE

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    child_index = 0

    # 第一步：把正文块聚合成父块
    buffer_texts: list[str] = []
    buffer_page: int | None = None
    buffer_section: str | None = None
    buffer_type = "text"

    def flush_buffer() -> None:
        nonlocal buffer_texts, buffer_page, buffer_section, buffer_type, child_index
        if not buffer_texts:
            return
        # 父块内部切子块：用 child_separators（不含单换行），保证"一条条款不被切碎"
        child_separators = _CHILD_SEPARATORS
        parent_text = "\n".join(buffer_texts)
        # 把章节标题作为父块的前缀写入内容：
        #   子块如果是"年假为5天"，脱离标题后语义是不完整的；
        #   父块带上"第二章 年假标准"后，喂给大模型时上下文才清楚，
        #   评测时也能看出这段内容属于哪一章（引用展示更友好）。
        if buffer_section and not parent_text.startswith(buffer_section):
            parent_text = f"{buffer_section}\n{parent_text}"
        p_index = len(parents)
        parents.append(
            ParentChunk(parent_index=p_index, content=parent_text,
                        page_no=buffer_page, section=buffer_section, block_type=buffer_type)
        )
        # 父块内部切子块
        if strategy == "semantic":
            # semantic_split 是异步的，这里先用递归兜底，真正的语义切分在主循环后统一重建
            sub_texts = recursive_split(parent_text, child_size, settings.CHUNK_OVERLAP,
                                        child_separators)
        else:
            sub_texts = recursive_split(parent_text, child_size, settings.CHUNK_OVERLAP,
                                        child_separators)
        for st in sub_texts:
            children.append(
                ChildChunk(
                    chunk_uid=new_chunk_uid(document_id, version, child_index),
                    content=st,
                    chunk_index=child_index,
                    parent_index=p_index,
                    page_no=buffer_page,
                    section=buffer_section,
                    block_type=buffer_type,
                    token_count=len(st),
                )
            )
            child_index += 1
        buffer_texts = []
        buffer_page = None
        buffer_section = None
        buffer_type = "text"

    for block in blocks:
        if block.is_empty:
            continue
        if block.block_type == "title":
            # 标题：结束上一个父块，并更新当前章节上下文
            flush_buffer()
            buffer_section = block.section or block.text
            continue

        current_len = sum(len(t) for t in buffer_texts)
        # 表格单独成父块：混进正文会污染语义
        if block.block_type == "table" or current_len + len(block.text) > parent_size:
            flush_buffer()
            buffer_section = block.section or buffer_section

        if not buffer_texts:
            buffer_page = block.page_no
            buffer_section = block.section or buffer_section
            buffer_type = block.block_type
        buffer_texts.append(block.text)

    flush_buffer()

    # 语义策略：对每个父块重新按语义切子块（需要异步调用 embedding）
    if strategy == "semantic" and children:
        logger.info("使用语义切分策略（会调用 embedding 接口）")
        rebuilt: list[ChildChunk] = []
        idx = 0
        for parent in parents:
            sub_texts = await semantic_split(parent.content, child_size, settings.SEMANTIC_THRESHOLD)
            for st in sub_texts:
                rebuilt.append(
                    ChildChunk(
                        chunk_uid=new_chunk_uid(document_id, version, idx),
                        content=st, chunk_index=idx,
                        parent_index=parent.parent_index, page_no=parent.page_no,
                        section=parent.section, block_type=parent.block_type, token_count=len(st),
                    )
                )
                idx += 1
        children = rebuilt

    result = ChunkResult(parents=parents, children=children)
    logger.info(f"切分完成 strategy={strategy} 统计={result.stats}")
    return result
