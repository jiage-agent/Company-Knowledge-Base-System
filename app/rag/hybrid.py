"""
混合检索编排 —— 本项目的技术核心之一。

完整链路（每一步都有存在理由，面试就按这个顺序讲）：

    查询
     ├─ 稠密检索（Chroma 向量）      -> 语义近似，抗同义改写
     └─ 稀疏检索（BM25 关键词）      -> 精确词命中，抗专有名词/编号
              ↓
         RRF 融合（按名次，免疫量纲差异 + 权重可配）
              ↓
         权限兜底过滤（defense in depth，向量库过滤之外再拦一道）
              ↓
         Cross-Encoder 重排（对 query-doc 逐对精算相关性，取 Top-N）
              ↓
         MMR 去冗余（避免 Top-5 全是同一段重复内容，浪费上下文窗口）
              ↓
         父块回填（小块召回、大块生成 —— 解决"召回准 vs 上下文全"的矛盾）

为什么重排必须放在融合之后（面试延伸）：
    Cross-Encoder 要对每一对 (query, chunk) 做一次完整的前向计算，
    成本随候选数线性增长。所以先用便宜的双塔向量检索把候选从百万级压到几十条，
    再对这几条做昂贵的精排 —— 这是"粗排 -> 精排"的标准漏斗结构。
    直接把全库丢给重排模型，延迟会无法接受。
"""

from __future__ import annotations

import asyncio
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.core.logging import get_logger
from app.core.permissions import PermissionContext, RetrievedChunk
from app.models import DocumentChunk
from app.rag.embedding import get_embedder
from app.rag.fusion import rrf_fuse_union
from app.rag.reranker import build_reranker
from app.rag.sparse import BM25Retriever
from app.vectorstore.base import get_vector_store

logger = get_logger(__name__)


# ============================================================
# 混合检索器
# ============================================================
class HybridRetriever:
    """稠密 + 稀疏 + RRF + 重排 + 父子块回填。"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.bm25 = BM25Retriever(session)
        self.reranker = build_reranker()

    async def retrieve(
        self,
        query: str,
        perm: PermissionContext,
        top_k: int | None = None,
        mode: str = "hybrid",
        where_extra: dict | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        执行检索。

        mode 用于消融实验（评测脚本会分别跑这三种配置）：
            dense_only        只用向量
            dense_bm25        向量 + BM25 + RRF
            dense_bm25_rerank 再加 Cross-Encoder 重排（默认，效果最好）

        返回 (结果列表, 调试信息)。调试信息会进日志和响应体，
        让"它为什么召回这几条"变得可解释 —— 排障和面试演示都需要。
        """
        t0 = time.perf_counter()
        top_k = top_k or settings.RERANK_TOP_K
        debug: dict = {"mode": mode, "query": query, "stages": {}}

        embedder = await get_embedder()
        store = await get_vector_store()

        # ---------- 第 1 步：稠密检索（带降级）----------
        # 降级策略：embedding 服务不可用（超时/限流/额度耗尽）时，
        # 不抛 500 让整个问答挂掉，而是**退化为纯关键词检索**，
        # 并在 debug 里标记 degraded=True，让调用方知道"这次结果质量打折了"。
        #
        # 为什么这个降级必须有（面试可讲）：
        #   向量化是问答链路上的外部依赖，它挂了不代表知识库挂了。
        #   用户宁可拿到"基于关键词命中的、质量略低但可用的答案"，
        #   也不愿意看到"服务不可用"。
        #   这也是"优雅降级"和"硬失败"的分界线 —— 生产系统的基本素养。
        degraded = False
        degrade_reason = ""
        try:
            qvec = await embedder.embed_query(query)
            dense = await store.search(
                qvec, settings.DENSE_TOP_K, perm, where_extra=where_extra
            )
        except Exception as e:  # noqa: BLE001
            degraded = True
            degrade_reason = f"embedding 不可用，已降级为纯关键词检索: {str(e)[:120]}"
            logger.warning(degrade_reason)
            dense = []
        debug["stages"]["dense"] = len(dense)
        if degraded:
            debug["degraded"] = True
            debug["degrade_reason"] = degrade_reason

        if mode == "dense_only":
            results = self._threshold_filter(dense, debug, settings.MIN_RELEVANCE_SCORE_DENSE)[:top_k]
            debug["stages"]["final"] = len(results)
            debug["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
            return await self._expand_parents(results, perm), debug

        # ---------- 第 2 步：稀疏检索（与稠密并行，降低总延迟）----------
        sparse_task = asyncio.create_task(
            self.bm25.search(query, settings.SPARSE_TOP_K, perm)
        )
        sparse = await sparse_task
        debug["stages"]["sparse"] = len(sparse)

        # ---------- 第 3 步：RRF 融合（并集语义）----------
        # 用 rrf_fuse_union 而不是 rrf_fuse：候选池必须是"稠密 ∪ 稀疏"。
        # 否则稀疏的新增候选会被截断掉，混合检索退化成"替换"而不是"补充"——
        # 实测症状就是"开了混合检索反而掉分"（Recall 0.525 -> 0.500）。
        # 真正的筛选交给第 5 步的重排（重排就是为筛候选而生的）。
        fused = rrf_fuse_union(
            [dense, sparse],
            weights=[settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT],
            k=settings.RRF_K,
            cap=max(settings.FUSION_TOP_K, settings.DENSE_TOP_K + settings.SPARSE_TOP_K),
        )
        debug["stages"]["fused"] = len(fused)

        # ---------- 第 4 步：权限兜底过滤 ----------
        before = len(fused)
        fused = [c for c in fused if perm.can_access(c.meta)]
        if len(fused) < before:
            logger.warning(f"权限兜底过滤拦截 {before - len(fused)} 条")
            debug["stages"]["permission_filtered"] = before - len(fused)

        if mode == "dense_bm25":
            # 这一组也做阈值过滤，保证三档配置的"拒答行为"一致，
            # 否则消融实验会把"阈值差异"算到"重排"头上。RRF 阈值默认关闭（见 config 注释）。
            results = self._threshold_filter(
                fused, debug, settings.MIN_RELEVANCE_SCORE_RRF)[:top_k]
            debug["stages"]["final"] = len(results)
            debug["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
            return await self._expand_parents(results, perm), debug

        # ---------- 第 5 步：Cross-Encoder 重排 ----------
        candidates = fused[: settings.RERANK_TOP_N]
        reranked = False
        if settings.RERANK_ENABLED and not degraded:
            results = await self.reranker.rerank(query, candidates, top_k)
            debug["stages"]["reranked"] = len(results)
            reranked = True
        else:
            results = candidates[:top_k]

        # ---------- 第 5.5 步：相关性阈值过滤（分阶段标定，见方法注释）----------
        # 阈值必须跟着"当前分数是谁打的"走（这是个容易忽略的坑）：
        #   重排成功 -> 用重排标定的阈值（0.115）
        #   重排关闭/降级 -> 候选上还是 RRF 分数，必须用 RRF 阈值（0.0 = 不过滤）
        # 如果重排没生效却套用重排阈值 0.115，而 RRF 分数只有 0.016，
        # 结果会被**全部过滤掉**、返回空 —— 这类"阈值用错场景"的 bug 很隐蔽。
        if reranked:
            results = self._threshold_filter(
                results, debug, settings.MIN_RELEVANCE_SCORE_RERANK)
        elif degraded:
            # 降级为纯关键词：BM25 分数尺度完全不同于余弦相似度，不做阈值过滤
            debug["stages"]["threshold_skipped"] = "degraded_bm25_only"
        else:
            results = self._threshold_filter(
                results, debug, settings.MIN_RELEVANCE_SCORE_RRF)

        # ---------- 第 6 步：MMR 去冗余 ----------
        results = await self._mmr(query, results, top_k)

        # ---------- 第 7 步：父块回填 ----------
        results = await self._expand_parents(results, perm)
        debug["stages"]["final"] = len(results)
        debug["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"检索完成 mode={mode} dense={debug['stages'].get('dense')} "
            f"sparse={debug['stages'].get('sparse')} fused={debug['stages'].get('fused')} "
            f"final={len(results)} 耗时={debug['elapsed_ms']}ms"
        )
        return results, debug

    # ---------------- 相关性阈值过滤 ----------------
    def _threshold_filter(
        self, chunks: list[RetrievedChunk], debug: dict, threshold: float | None = None
    ) -> list[RetrievedChunk]:
        """
        丢掉相关性过低的块。

        为什么必须有这一步（实测发现的缺陷）：
            向量检索永远会返回 Top-K，库里没有相关内容时也不例外 ——
            问"帮我写首诗"，照样返回 5 条不相关的制度条款。
            这些噪声进 prompt 会干扰生成，还让模型误以为"有证据可以答"，
            最终表现为"该拒答时硬答"。加阈值后，"没有相关内容"真的返回空。

        ⚠️ 阈值必须**分阶段标定**（这是踩过坑之后的修正，面试值得讲）：
            不同阶段的分数不可比：
              稠密检索  -> 余弦相似度 0~1（无关 0.1~0.25，相关 > 0.4）
              RRF 融合  -> 1/(k+rank) 累加，天然只有 0.01~0.03
              重排      -> 归一化相关性 0~1（无关 < 0.02）
            最初用一个 0.02 的统一阈值，结果把 RRF 档几乎全部过滤掉、
            指标从 0.83 崩到 0.16。教训：**跨阶段共用一个阈值是错的设计**，
            要么做分数归一化，要么各自标定 —— 我们选后者（实现简单且可控）。
        """
        if threshold is None:
            # 兼容旧配置：如果显式配了 MIN_RELEVANCE_SCORE，就统一用它
            threshold = settings.MIN_RELEVANCE_SCORE
        if threshold <= 0 or not chunks:
            return chunks
        kept = [c for c in chunks if c.score >= threshold]
        dropped = len(chunks) - len(kept)
        if dropped:
            key = f"below_threshold_{threshold}"
            debug["stages"][key] = dropped
            logger.info(f"相关性阈值 {threshold} 过滤掉 {dropped} 条")
        return kept

    # ---------------- MMR 去冗余 ----------------
    async def _mmr(
        self, query: str, candidates: list[RetrievedChunk], top_k: int, lambda_: float = 0.7
    ) -> list[RetrievedChunk]:
        """
        Maximal Marginal Relevance：在"相关"与"多样"之间取平衡。

        为什么需要：
            重排后的 Top-5 很可能全是同一段内容的相邻块（尤其父子块 + 重叠切分时），
            把 5 个几乎相同的块塞进 prompt，等于浪费上下文窗口，
            还会让大模型误以为"多处都这么说"，放大幻觉。
        MMR 的公式：argmax [ λ·sim(q,d) - (1-λ)·max sim(d, d_selected) ]
            λ=1 只看相关，λ=0 只看多样，0.7 是常用的偏相关取值。
        """
        if len(candidates) <= max(top_k, 2):
            return candidates

        embedder = await get_embedder()
        texts = [c.content for c in candidates]
        try:
            vecs = await embedder.embed_documents(texts)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MMR 向量化失败，跳过去冗余: {e}")
            return candidates[:top_k]

        from app.rag.embedding import cosine_similarity

        selected: list[int] = [0]
        remaining = list(range(1, len(candidates)))
        while len(selected) < top_k and remaining:
            best_idx, best_score = None, float("-inf")
            for i in remaining:
                rel = candidates[i].score
                redundancy = max(cosine_similarity(vecs[i], vecs[j]) for j in selected)
                mmr = lambda_ * rel - (1 - lambda_) * redundancy
                if mmr > best_score:
                    best_idx, best_score = i, mmr
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [candidates[i] for i in selected]

    # ---------------- 父块回填 ----------------
    async def _expand_parents(
        self, chunks: list[RetrievedChunk], perm: PermissionContext
    ) -> list[RetrievedChunk]:
        """
        把命中子块的父块全文取回来，供大模型生成时使用。

        为什么在这里做而不是在检索时做：
            父块文本很大，如果参与向量检索会稀释语义；
            只有确定要送入大模型的那几条才回填父块，既省 IO 又保精度。
        安全提示：父块同样要过一遍权限校验（父块与子块密级一致，
        但代码里不假设这一点 —— 不信任隐含约定是防越权的基本习惯）。
        """
        parent_uids = [c.parent_uid for c in chunks if c.parent_uid]
        if not parent_uids:
            return chunks

        stmt = select(DocumentChunk).where(DocumentChunk.chunk_uid.in_(parent_uids))
        rows = (await self.session.execute(stmt)).scalars().all()
        parent_map = {r.chunk_uid: r for r in rows}

        for c in chunks:
            if not c.parent_uid:
                continue
            p = parent_map.get(c.parent_uid)
            if p is None:
                continue
            if not perm.can_access({
                "tenant_id": p.tenant_id, "security_level": p.security_level,
                "department": p.department or "", "allowed_roles": p.allowed_roles or [],
                "is_active": p.is_active,
            }):
                logger.warning(f"父块权限校验未通过，已剥离: {p.chunk_uid}")
                continue
            c.parent_content = p.content
            c.meta["parent_uid"] = p.chunk_uid
        return chunks

    def invalidate_cache(self) -> None:
        self.bm25.invalidate()


async def build_retriever(session: AsyncSession) -> HybridRetriever:
    return HybridRetriever(session)
