"""
基于合成语料自动生成评测题（可验证的 ground truth）。

为什么要自动生成（关键设计决策）：
    手写评测题有个致命问题——**我不知道答案在哪一块**，只能靠关键词反查，
    而关键词会命中多个块，导致 gold 集合不唯一（上一版就踩了这个坑，
    结果是"48/75 题三档配置全部满分"，指标毫无区分度）。

    自动生成的题能保证 ground truth 100% 准确：
      1. 从文档里找出**全局唯一的特征串**（编号、金额、比例、专有格式）；
      2. 用这个特征串构造提问（并带上它所在的主题上下文）；
      3. 答案源块**必然是**包含该特征串的那个块 —— 唯一且可验证。

为什么这批题能体现 BM25 的价值（这是本次扩语料的目的）：
    特征串（如 `SCM-年份-四位流水号`、`不超过450元`）在向量空间里区分度低
    （语义模型把它们都当成"某个编号/某个金额"），
    而 BM25 对精确字面匹配极其敏感。
    语料从 16 块扩到上千块后，稠密检索的 Top-20 不再包含全部内容，
    稀疏检索才有机会贡献增量 —— 这样才能真正测出混合检索的价值。

用法：
    .venv\\Scripts\\python.exe scripts/gen_eval_from_corpus.py --per-doc 1 --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import BASE_DIR  # noqa: E402

# 特征串抽取规则：优先"编号格式/金额/比例/时限"，这些是 BM25 的主场
#
# ⚠️ 刻意不抽"示例编号"（形如 CS-2025-0037）（实测发现的标注缺陷）：
#    示例编号的共同部分是 "-2025-0037"，这个片段在多篇文档里都出现，
#    于是"只出现在 1 篇文档"的唯一性过滤会被它绕过去 ——
#    token 本身唯一（CS-2025-0037 vs PRD-2025-0037），但它的**匹配片段**不唯一，
#    导致关键词反查能同时命中多篇文档，gold 又不唯一了。
#    所以只保留"编号格式串"（CS-年份-四位流水号），它的前缀 CS/PRD/SCDA 是部门专有的。
PATTERNS = [
    (re.compile(r"[A-Z]{2,6}-年份-四位流水号"), "编号格式"),
    # 金额与比例
    (re.compile(r"不超过\s*\d+(?:\.\d+)?\s*(?:元|万元|天|个工作日|小时|分钟|个月|%)"), "金额/时限"),
    (re.compile(r"\d+(?:\.\d+)?\s*%"), "比例"),
    # 具体天数/次数
    (re.compile(r"(?:满|超过|不少于|不超过)\s*\d+\s*(?:天|次|个月|个工作日)"), "时限"),
]


def extract_candidates(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pat, kind in PATTERNS:
        for m in pat.finditer(text):
            token = m.group().strip()
            if len(token) < 4:
                continue
            out.append((token, kind))
    return out


def main(args: argparse.Namespace) -> None:
    corpus_dir = BASE_DIR / "data" / "corpus"
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        print("没有语料，请先运行 scripts/generate_corpus.py")
        return

    print(f"读取 {len(files)} 篇语料...")
    docs: dict[str, str] = {}
    for f in files:
        docs[f.name] = f.read_text(encoding="utf-8")

    # 统计每个特征串出现在多少篇文档里，只保留**低频**的（区分度高）
    token_docs: dict[str, set[str]] = defaultdict(set)
    token_pos: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, text in docs.items():
        for token, kind in extract_candidates(text):
            token_docs[token].add(name)
            token_pos[token].append((name, kind))

    unique_tokens = {t: d for t, d in token_docs.items() if len(d) == 1}
    print(f"抽到候选特征串 {len(token_docs)} 个，其中只出现在 1 篇文档里的 {len(unique_tokens)} 个")

    items: list[dict] = []
    per_doc_count: Counter = Counter()
    used = set()

    for token, names in sorted(unique_tokens.items(), key=lambda kv: -len(kv[0])):
        name = next(iter(names))
        if per_doc_count[name] >= args.per_doc:
            continue
        if token in used:
            continue
        if len(items) >= args.limit:
            break

        text = docs[name]
        pos = text.find(token)
        if pos < 0:
            continue

        # 文档标题 = 首行（形如《客户成功部差旅与住宿报销标准》）
        title = text.splitlines()[0].strip().strip("《》")
        # 所在章节 = 该位置之前最近的一个"第X章"标题
        prefix = text[:pos]
        chapters = re.findall(r"第\s*[一二三四五六七八九十\d]+\s*章\s*[^\n]{0,24}", prefix)
        chapter = chapters[-1].strip() if chapters else ""
        # 所属条款编号（"第X条"），用于把问题问得更具体
        articles = re.findall(r"第\s*[一二三四五六七八九十百零\d]+\s*条", prefix)
        article = articles[-1].strip() if articles else ""
        # 特征串所在的那句原文（用于人工核对 gold 是否正确）
        clause_start = max(prefix.rfind("。"), prefix.rfind("\n"))
        clause_end = text.find("。", pos)
        clause = text[clause_start + 1: clause_end + 1 if clause_end > 0 else pos + 80].strip()

        # 构造"自然 + 可精确匹配"的提问：
        #   带上文档名与章节，让问题有上下文（大模型与重排都能用上）；
        #   带上特征串原样，保证 BM25 能精确命中，同时使 gold 唯一。
        scope = f"《{title}》" + (f"{chapter}" if chapter else "")
        if token.startswith(("不超过", "满", "超过", "不少于")):
            question = f"{scope}里规定，{token}的标准是什么？"
        elif re.fullmatch(r"\d+(?:\.\d+)?\s*%", token):
            question = f"{scope}中，{token}对应的是哪项规定？"
        elif "年份" in token or re.match(r"^[A-Z]{2,6}-", token):
            question = f"{scope}里，{token}这种编号的格式要求是什么？"
        else:
            question = f"{scope}中关于“{token}”是怎么规定的？"
        if article:
            question += f"（{article}）"

        items.append({
            "question": question,
            "intent": "factual",
            "gold_keywords": [token],
            "expect_refusal": False,
            "_source_file": name,
            "_token_kind": token_pos[token][0][1],
            "_clause": clause,
        })
        used.add(token)
        per_doc_count[name] += 1

    # 补充一批"同义改写"题（这些是稠密检索的主场，用于对照）
    print(f"生成 {len(items)} 道精确匹配题")

    out_path = BASE_DIR / "data" / "eval_from_corpus.json"
    out_path.write_text(json.dumps({
        "_meta": {
            "description": "从合成语料自动生成的评测题，ground truth 由唯一特征串保证",
            "count": len(items),
            "note": "这批题的特征串在向量空间区分度低、在字面匹配上区分度高，用于验证 BM25 的增量价值",
        },
        "items": items,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已保存: {out_path}")
    print("\n样例（前 6 条）:")
    for it in items[:6]:
        print(f"  Q: {it['question']}")
        print(f"     源文件: {it['_source_file']}  特征类型: {it['_token_kind']}")
        print(f"     所在条款: {it['_clause'][:96]}")
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从语料自动生成评测题")
    p.add_argument("--per-doc", type=int, default=1, help="每篇文档最多出几道题")
    p.add_argument("--limit", type=int, default=60, help="总题数上限")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
