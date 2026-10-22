"""
中文分词 与 RRF 融合 —— 混合检索的两个基础零件。

一、为什么中文必须自己处理分词（面试可讲）：
    英文按空格切词即可；中文没有空格，"员工年假制度" 到底切成
    "员工/年假/制度" 还是 "员/工/年/假/制/度"，直接决定 BM25 的效果。
    工业界通常上 jieba 或 BERT 分词器，但这里用「字 + 二元组（bigram）」方案：
      "员工年假" -> 员,工,年,假, 员工,工年,年假
    理由：
      1. 零依赖、零加载耗时（jieba 首次加载字典要几百毫秒~秒级）；
      2. 对 BM25 这类词频统计模型，bigram 已经能捕捉"年假""制度"这类关键词，
         召回率接近专业分词器；
      3. 不会因为未登录词（公司内部黑话、产品代号）而切错。
    这是"效果/成本"权衡下的合理选择，也说明我不是照抄教程。

二、为什么需要 RRF 而不是简单加权求和（面试高频）：
    稠密检索的分数是余弦相似度（0~1 的连续值），
    稀疏检索 BM25 的分数是"无上界的对数概率"（可能是 12.7、也可能是 3.2），
    两者量纲完全不同。直接相加/加权会导致某一方被数值大小主导。
    RRF（Reciprocal Rank Fusion）只用「排名」不用「分数」：
        score(d) = Σ  1 / (k + rank_i(d))      k 常取 60
    好处：
      1. 免疫量纲差异，天然可融合任意多路召回；
      2. 对个别路的异常高分不敏感（鲁棒），因为只看名次；
      3. 无需调参（k=60 是论文推荐的经验值），而加权求和需要反复调权重。
    代价：丢掉了"分数差距"的信息（第1名和第2名差距多大不知道）。
"""

from __future__ import annotations

import re

from app.core.permissions import RetrievedChunk

# 英文/数字：连续字母数字作为一个 token
_ALNUM = re.compile(r"[a-zA-Z0-9]+")
# 中文（含日韩）：逐字
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def tokenize(text: str) -> list[str]:
    """
    中英混合分词：英文/数字按整词，中文按「单字 + 相邻二元组」。

    单字保证高召回（不会漏），二元组保证精度（"年假"比"年"+"假"更有区分度）。
    """
    if not text:
        return []
    tokens: list[str] = [m.group().lower() for m in _ALNUM.finditer(text)]

    cjk_chars = _CJK.findall(text)
    tokens.extend(cjk_chars)
    # 相邻二元组：捕获"年假""制度""报销"这类双字关键词
    for a, b in zip(cjk_chars, cjk_chars[1:]):
        tokens.append(a + b)
    return tokens


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion：把多路召回结果按排名融合成一路。

    参数：
        ranked_lists: 多路召回结果，每路已按各自的相关度降序排列
                      （例如 [稠密结果, 稀疏结果]）
        weights:      每路权重（默认等权）。稠密/稀疏权重可配是消融实验的调节旋钮。
        k:            平滑常数，越大则"名次差异"的影响越平缓，60 是论文经验值
        top_k:        融合后保留多少条

    实现要点：
        同一个 chunk 可能被多路同时召回（这恰恰是它重要的信号），
        所以用 chunk_uid 做 key 累加分数，而不是去重丢弃。
    """
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights 数量必须与召回路数一致")

    scores: dict[str, float] = {}
    store: dict[str, RetrievedChunk] = {}
    hits: dict[str, int] = {}

    for w, results in zip(weights, ranked_lists):
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_uid] = scores.get(item.chunk_uid, 0.0) + w * (1.0 / (k + rank))
            hits[item.chunk_uid] = hits.get(item.chunk_uid, 0) + 1
            # 保留信息更全的那一份（优先带 parent_content 的）
            if item.chunk_uid not in store or (
                item.parent_content and not store[item.chunk_uid].parent_content
            ):
                store[item.chunk_uid] = item

    merged: list[RetrievedChunk] = []
    for uid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        item = store[uid]
        item.score = score
        item.source = "fused"
        item.meta["rrf_score"] = score
        item.meta["multi_route_hits"] = hits[uid]     # 被几路同时命中，可解释性强
        merged.append(item)
    return merged


def rrf_fuse_union(
    ranked_lists: list[list[RetrievedChunk]],
    weights: list[float] | None = None,
    k: int = 60,
    cap: int | None = None,
) -> list[RetrievedChunk]:
    """
    RRF 融合（并集语义）：分数只用于排序，**不用于截断候选池**。

    为什么必须有这个版本（实测踩到的问题，面试很值得讲）：
        如果融合后也截断到 20 条（和单路 DENSE_TOP_K 一样大），
        融合就变成"用稀疏结果**替换**稠密结果"，而不是"补充"——
        稀疏带来的每条噪声都会挤掉一条稠密的好结果。
        实测症状：开了混合检索反而掉分（Recall 0.525 -> 0.500）。
        混合检索的价值来自"两路互补"，所以候选池必须是**并集**：
        稠密 20 条 + 稀疏里那些稠密没找到的（可能还有好几条）。
        真正的筛选交给下游的重排做（重排本来就是干这个的）。

    cap 只是内存保护，默认不启用；设置了也要远大于单路 top_k。
    """
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights 数量必须与召回路数一致")

    scores: dict[str, float] = {}
    store: dict[str, RetrievedChunk] = {}
    hits: dict[str, int] = {}
    origin: dict[str, list[str]] = {}
    route_names = ["dense", "sparse"] + [f"route{i}" for i in range(2, len(ranked_lists))]

    for route, (w, results) in enumerate(zip(weights, ranked_lists)):
        name = route_names[route] if route < len(route_names) else f"route{route}"
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_uid] = scores.get(item.chunk_uid, 0.0) + w * (1.0 / (k + rank))
            hits[item.chunk_uid] = hits.get(item.chunk_uid, 0) + 1
            origin.setdefault(item.chunk_uid, []).append(name)
            if item.chunk_uid not in store or (
                item.parent_content and not store[item.chunk_uid].parent_content
            ):
                store[item.chunk_uid] = item

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if cap:
        ordered = ordered[:cap]

    merged: list[RetrievedChunk] = []
    for uid, score in ordered:
        item = store[uid]
        item.score = score
        item.source = "fused"
        item.meta["rrf_score"] = score
        item.meta["multi_route_hits"] = hits[uid]
        item.meta["routes"] = origin[uid]
        merged.append(item)
    return merged
