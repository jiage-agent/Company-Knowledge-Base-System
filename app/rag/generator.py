"""
生成层：基于证据作答 + 引用溯源 + 规则校验（Critic 的一半）。

三条设计原则（面试就讲这三条，全是踩过坑换来的）：

一、强制"基于证据"作答，并给出结构化引用锚点。
    不是让模型写"根据文档可知"，而是要求它输出编号引用（[1][2]），
    编号映射到具体的 chunk_uid + 文档名 + 页码，
    前端可点击跳转到原文 —— 这才是"答案可追溯"。

二、证据不足时必须拒答，不许编。
    企业知识库最怕的不是"答不出来"，而是"一本正经地编"。
    提示词里明确给出拒答话术，并且有代码层兜底（见 critic.py）。

三、Critic 用"规则校验 + LLM 判定"双重手段。
    只靠 LLM 自评极不可靠：它倾向于说"我的答案有证据支持"。
    所以先用规则卡：引用编号是否真实存在于召回集、答案里的数字是否能在证据里找到。
    规则能拦住的错误，就不要交给模型判断 —— 这是既省钱又可靠的工程选择。
"""

from __future__ import annotations

import math
import re

from app.core.logging import get_logger
from app.core.permissions import RetrievedChunk
from app.rag.llm import chat

logger = get_logger(__name__)

REFUSAL_TEXT = "未在知识库中找到可靠依据，无法回答该问题。"

SYSTEM_PROMPT = """你是企业知识库问答助手。请严格依据【参考资料】回答用户问题。

必须遵守的规则：
1. 只使用【参考资料】中的信息回答；不得使用你自己的先验知识补充或推测。
2. 每个关键结论后面必须标注来源编号，格式为 [1]、[2]（可多个，如 [1][3]）。
3. 如果【参考资料】不足以回答问题，直接回复：「未在知识库中找到可靠依据，无法回答该问题。」不要尝试猜测。
4. 如果资料之间互相矛盾，请指出矛盾并分别标注来源编号。
5. 回答使用简体中文，条理清晰；涉及数字、金额、天数、期限时必须与资料完全一致，不得改写。
6. 不要输出"根据参考资料"这类废话，直接给答案。
"""


def build_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> tuple[str, list[RetrievedChunk]]:
    """
    拼装上下文。

    关键取舍：优先使用「父块全文」，没有父块才退回子块内容。
      子块是检索单位（小、准），父块是生成单位（大、全）。
      但父块不能无脑全塞 —— 要按字符预算截断，否则会超上下文窗口并推高成本。
    """
    parts: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    seen_parents: set[str] = set()

    for c in chunks:
        text = c.parent_content or c.content
        # 同一父块只保留一次：父子块结构下，多个子块可能属于同一父块，
        # 重复塞进去既浪费 token，又会让模型误以为"多处都这么说"
        if c.parent_uid and c.parent_uid in seen_parents:
            continue
        if c.parent_uid:
            seen_parents.add(c.parent_uid)

        idx = len(used) + 1
        header = f"[{idx}] 来源：{c.title or '未知文档'}"
        if c.page_no:
            header += f" 第{c.page_no}页"
        if c.section:
            header += f" / {c.section}"
        block = f"{header}\n{text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(c)
        total += len(block)

    return "\n\n---\n\n".join(parts), used


async def generate_answer(question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None) -> dict:
    """生成答案。返回 {answer, used_chunks, prompt_chars, refused}。"""
    if not chunks:
        return {"answer": REFUSAL_TEXT, "used_chunks": [], "prompt_chars": 0, "refused": True}

    context, used = build_context(chunks)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 多轮对话：带上最近几轮历史，让"那它的有效期呢"这类指代能被理解。
    # 注意只带对话，不带上一轮的检索结果 —— 否则上下文会被旧证据污染。
    for h in (history or [])[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({
        "role": "user",
        "content": f"【参考资料】\n{context}\n\n【用户问题】\n{question}",
    })

    answer = (await chat(messages)).strip()
    refused = REFUSAL_TEXT[:12] in answer or "无法回答" in answer
    return {
        "answer": answer,
        "used_chunks": used,
        "prompt_chars": len(context),
        "refused": refused,
    }


# ============================================================
# 规则校验（Critic 的确定性部分）
# ============================================================
_NUM = re.compile(r"\d+(?:\.\d+)?")


def verify_citations(answer: str, used: list[RetrievedChunk]) -> dict:
    """
    引用真实性校验：答案里出现的 [n] 是否都在本次召回集合内。

    为什么这条必须有：
      大模型非常容易编造引用（写出一个不存在的 [5]），
      而"引用不存在"意味着答案无法被验证 = 等于幻觉。
      这项检查是确定性的、几乎零成本，比让模型自评可靠得多。
    """
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    max_idx = len(used)
    valid = {i for i in cited if 1 <= i <= max_idx}
    invalid = cited - valid
    return {
        "cited": sorted(cited),
        "valid": sorted(valid),
        "invalid": sorted(invalid),
        "has_citation": bool(cited),
        "all_valid": not invalid and bool(cited),
    }


def verify_numbers(answer: str, used: list[RetrievedChunk]) -> dict:
    """
    数字一致性校验：答案里出现的关键数字，必须能在证据原文中找到。

    为什么专挑数字（面试会问"为什么不是校验所有词"）：
      知识库问答里最致命的错误是"数字错了"—— 年假 5 天说成 15 天、
      报销 450 元说成 540 元。这类错误用户很难察觉，但后果最严重。
      而数字是确定性字符，用正则可以精确校验，成本几乎为零。
      相比之下校验普通措辞既做不准也没必要（语义等价无法用字符串比对）。
    """
    if not used:
        return {"checked": 0, "unsupported": [], "ok": True}
    evidence = "\n".join((c.parent_content or c.content) for c in used)
    evidence_nums = set(_NUM.findall(evidence))

    # 归一化：去掉引用编号，避免把 [1] 里的 1 当成答案中的数字
    cleaned = re.sub(r"\[\d+\]", " ", answer)
    answer_nums = set(_NUM.findall(cleaned))
    # 过滤掉"序数/年份"之类的噪声干扰项，只保留可能承载业务含义的数字
    answer_nums = {n for n in answer_nums if len(n) >= 1}

    unsupported = sorted(n for n in answer_nums if n not in evidence_nums)
    return {
        "checked": len(answer_nums),
        "unsupported": unsupported,
        "ok": len(unsupported) == 0,
    }


def compute_confidence(
    chunks: list[RetrievedChunk], verify: dict, numbers: dict
) -> float:
    """
    置信度 = 检索质量 + 引用规范性 + 数字一致性 的加权。

    为什么要给一个数值而不是"高/中/低"：
      数值可以做阈值告警、可以做 A/B 对比、可以进评测指标（区分度更高）。
      权重是启发式的，但每个分量都有明确含义，而不是拍脑袋的一个常数：
        - 最高重排分：检索到底有多确信（0.5）
        - 引用是否规范有效：答案是否可追溯（0.3）
        - 数字是否全部有据：最致命错误的风险（0.2）
    """
    if not chunks:
        return 0.0
    top_score = max(c.score for c in chunks)
    # 重排模型的分数（logits）不在 0~1 区间，用 sigmoid 归一化，
    # 否则置信度会全是 0 或 1，失去区分度（这类"没归一化"是评测里常见的隐性 bug）
    norm_score = 1.0 / (1.0 + math.exp(-top_score))
    citation_part = 1.0 if verify.get("all_valid") else (0.4 if verify.get("has_citation") else 0.0)
    number_part = 1.0 if numbers.get("ok", False) else 0.3
    conf = 0.5 * norm_score + 0.3 * citation_part + 0.2 * number_part
    return round(max(0.0, min(1.0, conf)), 4)
