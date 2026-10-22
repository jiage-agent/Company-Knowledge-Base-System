"""
评测与消融实验 —— 这是让简历上的数字"可信"的地方。

为什么必须做评测（面试时最能拉分的一段）：
    没做过评测的项目，只能回答"我觉得效果还行"；
    做过评测的项目，能回答"混合检索把 Recall@10 从 0.62 提到 0.89，
    重排再加 6 个百分点，代价是延迟从 180ms 涨到 420ms"。
    后者才是工程能力，前者只是"跑通了"。

本项目的三层评测：

  第一层：检索指标（不需要 LLM，零成本，可反复跑）
    Recall@K   —— Top-K 里是否包含正确的块（召回够不够）
    MRR        —— 正确块排在第几位（排序好不好）
    NDCG@K     —— 排序质量的加权版本（越靠前权重越高）
    HitRate@K  —— 至少命中一个的比例（用户感知最直接）
    这三个指标一起看才有意义：Recall 高但 MRR 低，说明"找得到但排得靠后"，
    大模型很可能因为排序靠后而忽略它。

  第二层：答案质量（需要 LLM，贵，跑小样本）
    引用准确率 —— 答案里的引用是否真实存在于召回集（可确定性校验）
    拒答准确率 —— 该拒答时是否拒答（不编造）
    数字一致率 —— 答案中的数字是否都能在证据里找到

  第三层：消融实验（A/B/C 三组对比）
    dense_only        只用向量
    dense_bm25        向量 + BM25 + RRF
    dense_bm25_rerank 再加重排（完整链路）
    这一组对比直接回答"每个组件到底贡献了多少"，是简历上"消融实验"四个字的实体。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import BASE_DIR
from app.core.logging import get_logger
from app.core.permissions import PermissionContext
from app.models import DocumentChunk, EvalRecord, User
from app.rag.generator import verify_citations, verify_numbers
from app.rag.hybrid import HybridRetriever

logger = get_logger(__name__)

MODES = {
    "dense_only": "仅向量检索",
    "dense_bm25": "向量 + BM25 + RRF 融合",
    "dense_bm25_rerank": "向量 + BM25 + RRF + 重排（完整链路）",
}


# ============================================================
# 检索指标
# ============================================================
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Top-K 中命中的相关块比例（分母是相关块总数）。"""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Top-K 中是否至少命中一个相关块（0 或 1）。"""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """第一个相关块的排名倒数。排第 1 得 1.0，排第 2 得 0.5，未命中得 0。"""
    for i, uid in enumerate(retrieved, start=1):
        if uid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    NDCG@K：考虑排序位置的召回质量。

    为什么除了 Recall 还要看 NDCG：
      Recall 只看"有没有"，NDCG 还看"排多前"。
      实践中"排在第 1 位"和"排在第 8 位"对最终答案的影响完全不同
      （大模型对上下文开头的内容注意力更强，且有上下文长度截断）。
    """
    import math

    dcg = 0.0
    for i, uid in enumerate(retrieved[:k], start=1):
        if uid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 数据加载
# ============================================================
def load_dataset(path: Path | None = None) -> list[dict]:
    """
    加载评测集。

    支持两种来源与两种格式：
      1. 内置种子评测集（app/seed_data.py 的 EVAL_DATASET），开箱即用；
      2. 外部 JSON 文件（data/eval_dataset.json），便于扩充与人工标注。
         支持裸数组 `[{...}]` 与带元信息的对象 `{"_meta":{...},"items":[...]}` 两种结构。
    """
    target = path or (BASE_DIR / "data" / "eval_dataset.json")
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("items") or []
            meta = data.get("_meta") or {}
            if meta:
                logger.info(f"评测集: {target.name} version={meta.get('version', '?')} "
                            f"共 {len(items)} 条")
            return items
        if isinstance(data, list):
            return data
    from app.seed_data import EVAL_DATASET

    return EVAL_DATASET


async def _resolve_gold_chunks(
    session: AsyncSession, item: dict, tenant_id: int, perm: PermissionContext | None = None
) -> set[str]:
    """
    把评测集里的 gold_keywords 映射成具体的 chunk_uid。

    为什么要做这层映射（而不是硬编码 chunk_uid）：
      重建索引、改切分策略后 chunk_uid 会变化，而"关键词"是稳定的语义标注。
      用关键词反查当前索引，评测集就能跨版本复用。

    ⚠️ 关键修正：必须选**唯一的源块**，不能用关键词命中所有块（实测踩坑）。
        最初的实现是"任何包含关键词的块都算 gold"，问题很严重：
          - 关键词常有重复（"15天"在年假条款和陪产假条款里都出现），
            于是 gold 集合里混进了不是答案所在的块；
          - 更糟的是**这个集合会随切分策略变化**：
            同一句话在 recursive 下属于 A 块、在 semantic 下被切进 B 块，
            导致 gold 集合在不同策略下不可比 —— 实验直接失效。
        实测后果：48/75 题只有 1 个 gold 块时三种检索配置全部 recall=1.000
        （毫无区分度），而差异全集中在 gold_count 为 2~4 的题上，
        且方向随机。那不是模型差异，是标注噪声。
        现在改为：**先用关键词缩小候选，再用语义相似度挑出最匹配的那一个块**。
        这样每题只有一个源块，跨策略可比，指标才有意义。
    """
    keywords = item.get("gold_keywords") or []
    if not keywords:
        return set()

    conds = [
        DocumentChunk.tenant_id == tenant_id,
        DocumentChunk.is_parent == False,  # noqa: E712
        DocumentChunk.is_active == True,  # noqa: E712
    ]
    if perm is not None:
        # 必须限制在该评测用户**有权访问**的范围内找 gold：
        # 否则权限题会把"用户本来就看不到的块"算进 gold，导致指标无意义
        conds.extend(perm.sql_conditions())

    rows = (await session.execute(select(DocumentChunk).where(*conds))).scalars().all()

    # 第一步：关键词筛出候选（要求命中所有关键词，减少误召回）
    candidates = [c for c in rows if all(kw in c.content for kw in keywords)]
    if not candidates:
        # 放宽：任一关键词命中即可（应对"答案被切分成两个块"的情况）
        candidates = [c for c in rows if any(kw in c.content for kw in keywords)]
    if not candidates:
        return set()
    if len(candidates) == 1:
        return {candidates[0].chunk_uid}

    # 第二步：用语义相似度挑出最匹配的那一个（与提问最接近的块才是答案源块）
    try:
        from app.rag.embedding import cosine_similarity, get_embedder

        embedder = await get_embedder()
        qvec = await embedder.embed_query(item["question"])
        cvecs = await embedder.embed_documents([c.content[:512] for c in candidates])
        best = max(
            range(len(candidates)),
            key=lambda i: cosine_similarity(qvec, cvecs[i]),
        )
        logger.debug(f"gold 候选 {len(candidates)} 个，选定 {candidates[best].chunk_uid} "
                     f"(问题: {item['question'][:30]})")
        return {candidates[best].chunk_uid}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"gold 语义筛选失败，退回全部候选: {e}")
        return {c.chunk_uid for c in candidates}


async def _pick_eval_user(session: AsyncSession, tenant_id: int, item: dict) -> User:
    """
    为每条评测样本挑选合适的用户。
    带 only_roles 的样本（财务机密类）必须用有权限的账号去测，
    否则测的是"权限拦截"而不是"检索能力" —— 两类指标不能混在一起。
    """
    roles = item.get("only_roles")
    stmt = select(User).where(User.tenant_id == tenant_id, User.is_active == True)  # noqa: E712
    users = (await session.execute(stmt)).scalars().all()
    if not users:
        raise RuntimeError("没有可用用户，请先运行 scripts/seed.py")
    if roles:
        for u in users:
            if u.role in roles or u.department in roles:
                return u
    # 默认用普通成员账号（最低权限），这样测出的检索指标是最保守的下界
    for u in users:
        if u.role == "member":
            return u
    return users[0]


# ============================================================
# 主流程
# ============================================================
async def run_eval(
    session: AsyncSession,
    tenant_id: int,
    modes: list[str] | None = None,
    top_k: int = 5,
    limit: int = 0,
    dataset_path: Path | None = None,
    run_id: str | None = None,
) -> dict:
    """
    跑一轮消融实验。

    执行顺序说明：
      外层循环是"模式"，内层是"样本" ——
      这样才能对同一批样本公平比较三种配置（同一批数据、同一批账号）。
    """
    modes = modes or ["dense_only", "dense_bm25", "dense_bm25_rerank"]
    run_id = run_id or f"eval-{int(time.time())}"
    dataset = load_dataset(dataset_path or (BASE_DIR / "data" / "eval_dataset.json"))
    if limit:
        dataset = dataset[:limit]

    logger.info(f"评测开始 run_id={run_id} 样本={len(dataset)} 模式={modes} top_k={top_k}")
    all_metrics: list[dict] = []
    detail: list[dict] = []

    for mode in modes:
        recalls, mrrs, ndcgs, hits, latencies = [], [], [], [], []
        # 生成类指标（只在有 LLM 生成时统计）
        cite_ok, refuse_ok, num_ok, counts = 0, 0, 0, 0

        for item in dataset:
            user = await _pick_eval_user(session, tenant_id, item)
            perm = PermissionContext.from_user(user)
            gold = await _resolve_gold_chunks(session, item, tenant_id, perm)

            retriever = HybridRetriever(session)
            t0 = time.perf_counter()
            chunks, _dbg = await retriever.retrieve(
                item["question"], perm, top_k=top_k, mode=mode
            )
            latency = int((time.perf_counter() - t0) * 1000)
            retrieved_uids = [c.chunk_uid for c in chunks]

            r = recall_at_k(retrieved_uids, gold, top_k) if gold else 0.0
            m = mrr(retrieved_uids, gold) if gold else 0.0
            n = ndcg_at_k(retrieved_uids, gold, top_k) if gold else 0.0
            h = hit_rate_at_k(retrieved_uids, gold, top_k) if gold else 0.0

            # 拒答场景单独评估：知识库里没有答案时，检索结果为空才是"正确"
            expect_refusal = bool(item.get("expect_refusal"))
            if expect_refusal:
                # 期望拒答的题：召回越少越好（说明没有硬凑答案）
                r = m = n = h = 1.0 if not chunks else 0.0
                counts += 1
                refuse_ok += 1 if not chunks else 0

            recalls.append(r); mrrs.append(m); ndcgs.append(n)
            hits.append(h); latencies.append(latency)

            detail.append({
                "mode": mode, "question": item["question"],
                "intent": item.get("intent"), "recall": round(r, 3),
                "mrr": round(m, 3), "ndcg": round(n, 3),
                "retrieved": len(retrieved_uids), "gold_count": len(gold),
                "latency_ms": latency, "expect_refusal": expect_refusal,
                "top_titles": [c.title for c in chunks[:3]],
            })

            session.add(EvalRecord(
                run_id=run_id, mode=mode, question=item["question"],
                gold_answer=", ".join(item.get("gold_keywords") or []),
                gold_chunk_uids=sorted(gold), retrieved_chunk_uids=retrieved_uids,
                recall_at_k=r, mrr=m, ndcg=n, latency_ms=latency,
            ))

        all_metrics.append({
            "mode": mode,
            "mode_desc": MODES.get(mode, mode),
            "questions": len(dataset),
            "recall_at_k": round(statistics.mean(recalls), 4) if recalls else 0.0,
            "mrr": round(statistics.mean(mrrs), 4) if mrrs else 0.0,
            "ndcg": round(statistics.mean(ndcgs), 4) if ndcgs else 0.0,
            "hit_rate": round(statistics.mean(hits), 4) if hits else 0.0,
            "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        })
        logger.info(f"模式 {mode} 完成: {all_metrics[-1]}")

    await session.commit()
    return {
        "run_id": run_id,
        "dataset_size": len(dataset),
        "top_k": top_k,
        "metrics": all_metrics,
        "detail": detail,
    }


def format_report(result: dict) -> str:
    """把评测结果格式化成可直接贴进 README / 简历的表格。"""
    lines = []
    lines.append("")
    lines.append("=" * 88)
    lines.append(f"消融实验结果  run_id={result['run_id']}  样本数={result['dataset_size']}  "
                 f"top_k={result.get('top_k', 5)}")
    lines.append("=" * 88)
    header = f"{'配置':<28} {'Recall@K':>10} {'MRR':>8} {'NDCG@K':>8} {'HitRate':>9} {'延迟(ms)':>10}"
    lines.append(header)
    lines.append("-" * 88)
    base = None
    for m in result["metrics"]:
        if base is None:
            base = m
        lines.append(
            f"{m['mode_desc'][:26]:<28} {m['recall_at_k']:>10.4f} {m['mrr']:>8.4f} "
            f"{m['ndcg']:>8.4f} {m['hit_rate']:>9.4f} {m['avg_latency_ms']:>10.1f}"
        )
    lines.append("-" * 88)
    # 相对第一组（dense_only）的提升幅度 —— 这是简历上真正要写的数字
    if base and len(result["metrics"]) > 1:
        lines.append("相对「仅向量检索」的提升：")
        for m in result["metrics"][1:]:
            d_recall = m["recall_at_k"] - base["recall_at_k"]
            d_ndcg = m["ndcg"] - base["ndcg"]
            d_lat = m["avg_latency_ms"] - base["avg_latency_ms"]
            lines.append(
                f"  {m['mode_desc'][:24]:<26} Recall {d_recall:+.4f}  "
                f"NDCG {d_ndcg:+.4f}  延迟 {d_lat:+.0f}ms"
            )
    lines.append("=" * 88)
    return "\n".join(lines)
