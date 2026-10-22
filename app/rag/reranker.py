"""
重排器（Reranker）—— 粗排之后的精排环节。

为什么需要重排（面试必答）：
    向量检索用的是「双塔」结构：query 和文档分别编码成向量，只算一次内积。
    好处是能预先建索引、检索快；代价是 query 和文档从未"见面"，
    细粒度的词级交互信息全部丢失，精度有天花板。
    重排用的是「交叉编码器」：把 (query, doc) 拼在一起送进模型，
    注意力机制可以逐词比对，精度显著更高 —— 但必须逐对计算，无法预建索引。
    所以标准做法是漏斗结构：双塔粗排（百万级 -> 几十条）+ 交叉精排（几十条 -> 几条）。

本项目提供三种实现，通过 RERANK_PROVIDER 热切换：
    api   —— 阿里云百炼 gte-rerank-v2。零下载、开箱可用，返回 0~1 相关性分数。
             实测：query"年假有几天" 对三条文档打分 0.381 / 0.134 / 0.006，
             排序正确，且单次调用仅 107 tokens，成本可忽略。
    local —— 本地 Cross-Encoder（BAAI/bge-reranker-base），离线可用、无 API 成本，
             但需要下载 1GB+ 模型，且受 transformers 版本兼容性影响（实测踩过坑）。
    none  —— 不重排，直接用融合结果。用于消融实验的对照组。

工程要点：重排失败必须【降级】而不是【报错】。
    重排是"锦上添花"的一环，它挂了不应该让整个问答不可用 ——
    降级为按融合分数排序，用户仍能拿到答案，只是精度略降。
    这种"优雅降级"是生产系统的基本素养。
"""

from __future__ import annotations

import asyncio
import time

from app.config import settings
from app.core.logging import get_logger
from app.core.permissions import RetrievedChunk

logger = get_logger(__name__)


class BaseReranker:
    name = "base"

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class DashScopeReranker(BaseReranker):
    """
    阿里云百炼文本重排（gte-rerank-v2）。

    接口格式与 OpenAI 的 /embeddings 不同，走 DashScope 原生路径：
        POST {base}/api/v1/services/rerank/text-rerank/text-rerank
        body: {"model":..., "input":{"query":..., "documents":[...]}, "parameters":{...}}
    注意：没有 /compatible-mode/v1/rerank 这个路径（实测 404），
    直接按官方原生协议调用才是对的 —— 这类"文档没写清"的接口差异很常见。
    """

    name = "dashscope-rerank"

    def __init__(self) -> None:
        base = settings.EMBEDDING_BASE_URL.rstrip("/")
        # 专属 host 与官方域名都支持原生路径，这里从 embedding 的 base_url 推导
        self.endpoint = base.replace("/compatible-mode/v1", "") + \
            "/api/v1/services/rerank/text-rerank/text-rerank"
        self.api_key = settings.EMBEDDING_API_KEY
        self.model = settings.RERANK_MODEL
        self._fail_count = 0          # 连续失败计数，用于熔断

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        # 熔断：连续失败 3 次就不再尝试，避免每次都白等超时（拖慢整体响应）
        if self._fail_count >= 3:
            logger.warning("重排接口连续失败，本次降级为不重排")
            return candidates[:top_k]

        import httpx

        payload = {
            "model": self.model,
            "input": {"query": query, "documents": [c.content[:1500] for c in candidates]},
            "parameters": {"return_documents": False, "top_n": min(top_k, len(candidates))},
        }
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    self.endpoint, json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
            results = (data.get("output") or {}).get("results") or []
            if not results:
                raise ValueError("重排接口返回空结果")

            ranked: list[RetrievedChunk] = []
            for item in results:
                idx = int(item["index"])
                if idx >= len(candidates):
                    continue
                c = candidates[idx]
                c.meta["pre_rerank_score"] = c.score     # 保留重排前分数，便于解释与对比
                c.score = float(item.get("relevance_score", 0.0))
                c.source = "rerank"
                ranked.append(c)
            self._fail_count = 0
            logger.info(f"重排完成 {len(candidates)} -> {len(ranked)} 条，"
                        f"耗时 {int((time.perf_counter() - t0) * 1000)}ms")
            return ranked[:top_k]
        except Exception as e:  # noqa: BLE001
            self._fail_count += 1
            logger.warning(f"重排失败（第 {self._fail_count} 次），降级为按融合分数排序: {e}")
            return candidates[:top_k]


class LocalCrossEncoderReranker(BaseReranker):
    """
    本地 Cross-Encoder 重排（离线场景）。

    模型懒加载 + 单例：加载一次权重（约 1GB）后常驻，
    绝不每次请求都 load —— 那会让单次问答延迟增加几秒。
    """

    name = "local-cross-encoder"

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._load_failed = False

    def _ensure_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder

            device = settings.RERANK_DEVICE
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            logger.info(f"加载本地重排模型 {settings.RERANK_LOCAL_MODEL} (device={device}) ...")
            self._model = CrossEncoder(settings.RERANK_LOCAL_MODEL, device=device, max_length=512)
            logger.info("本地重排模型加载完成")
        except Exception as e:  # noqa: BLE001
            self._load_failed = True
            logger.warning(f"本地重排模型加载失败，将跳过重排（降级运行）: {e}")
        return self._model

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._ensure_model()
        if model is None:
            return candidates[:top_k]

        pairs = [(query, c.content) for c in candidates]
        scores = await asyncio.to_thread(
            lambda: model.predict(pairs, batch_size=16, show_progress_bar=False)
        )
        for c, s in zip(candidates, scores):
            c.meta["pre_rerank_score"] = c.score
            c.score = float(s)
            c.source = "rerank"
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]


class NoopReranker(BaseReranker):
    """关闭重排（消融实验对照组）。"""

    name = "none"

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_k]


def build_reranker() -> BaseReranker:
    """按配置构造重排器（热切换点）。"""
    if not settings.RERANK_ENABLED:
        logger.info("重排已关闭（RERANK_ENABLED=false）")
        return NoopReranker()
    provider = settings.RERANK_PROVIDER.lower()
    if provider == "api":
        return DashScopeReranker()
    if provider == "local":
        return LocalCrossEncoderReranker()
    if provider in ("none", "off"):
        return NoopReranker()
    raise ValueError(f"未知的 RERANK_PROVIDER: {settings.RERANK_PROVIDER}")
