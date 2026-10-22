"""
Embedding 抽象层 —— "热切换向量模型"的落地点。

为什么要这一层（面试可讲）：
    检索、入库两处都要用向量，如果直接在业务代码里写死 openai/dashscope 的调用，
    换模型（阿里云 text-embedding-v4 -> 本地 BGE-M3）就得改多处代码。
    抽成接口后，切换只改 .env 的 EMBEDDING_PROVIDER，业务代码零改动。

本文件提供两个实现：
    DashScopeEmbedder —— 阿里云百炼 text-embedding-v4（1024 维），走 OpenAI 兼容协议，异步并发
    LocalEmbedder     —— 本地 sentence-transformers（BAAI/bge-small-zh-v1.5 等），离线可用

统一契约：embed_documents(texts) -> list[list[float]]、embed_query(text) -> list[float]
"""

from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """所有向量模型的统一接口。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库时用）。"""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """单条向量化（检索时用）。"""

    async def close(self) -> None:
        return None


class DashScopeEmbedder(BaseEmbedder):
    """
    阿里云百炼 Embedding（OpenAI 兼容协议）。

    工程细节：
    1. 批量 + 并发：接口单次最多 10 条文本（text-embedding-v4 的限制），
       所以既要分批又要用信号量控制并发，否则大批文档会超时或被限流。
    2. 失败重试：指数退避重试 3 次，网络抖动不至于让整篇文档入库失败。
    3. 文本清洗：接口不接受空字符串，空文本替换成占位符，保证下标对齐
       （下标错位是向量与块内容错配的典型 bug）。
    """

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self.name = settings.EMBEDDING_MODEL
        self.dim = settings.EMBEDDING_DIM
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.max_concurrency = 4
        self._client = AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=60.0,
            max_retries=0,          # 自己控制重试，便于记录日志
        )

    async def _embed_batch(self, batch: list[str], sem: asyncio.Semaphore) -> list[list[float]]:
        async with sem:
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    resp = await self._client.embeddings.create(
                        model=self.name,
                        input=batch,
                        dimensions=self.dim,   # text-embedding-v4 支持指定输出维度
                    )
                    # 接口不保证返回顺序，按 index 排序是稳妥做法
                    data = sorted(resp.data, key=lambda d: d.index)
                    return [d.embedding for d in data]
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    wait = 1.5 * (attempt + 1)
                    logger.warning(f"embedding 第{attempt + 1}次失败({e})，{wait}s 后重试")
                    await asyncio.sleep(wait)
            raise RuntimeError(f"embedding 连续失败: {last_err}")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clean = [(t.strip() or "空内容")[:6000] for t in texts]
        batches = [clean[i:i + self.batch_size] for i in range(0, len(clean), self.batch_size)]
        sem = asyncio.Semaphore(self.max_concurrency)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(self._embed_batch(b, sem) for b in batches))
        vecs = [v for group in results for v in group]
        cost = time.perf_counter() - t0
        logger.info(f"向量化 {len(texts)} 条 / {len(batches)} 批，耗时 {cost:.2f}s "
                    f"({len(texts) / cost:.1f} 条/秒)")
        return vecs

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self.embed_documents([text])
        return vecs[0]

    async def close(self) -> None:
        await self._client.close()


class LocalEmbedder(BaseEmbedder):
    """
    本地 sentence-transformers 实现（离线场景 / 省钱场景）。

    模型在首次调用时懒加载：不 import 就不占内存，
    避免只用云端接口时白白加载 torch。
    """

    def __init__(self) -> None:
        self.name = settings.EMBEDDING_LOCAL_MODEL
        self._model = None
        self._device = "cpu"

    def _ensure_model(self):
        if self._model is None:
            import os

            os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)  # 国内镜像
            import torch
            from sentence_transformers import SentenceTransformer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"加载本地 embedding 模型 {self.name} (device={self._device}) ...")
            self._model = SentenceTransformer(self.name, device=self._device)
            self.dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成，维度={self.dim}")
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()

        def _run() -> list[list[float]]:
            embs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                                show_progress_bar=False)
            return [e.tolist() for e in embs]

        # sentence-transformers 是同步阻塞的，丢到线程池避免卡住事件循环
        return await asyncio.to_thread(_run)

    async def embed_query(self, text: str) -> list[float]:
        vecs = await self.embed_documents([text])
        return vecs[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度。用于语义切分判断相邻句是否该断开。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


# ---------------- 工厂 + 单例 ----------------
_embedder: BaseEmbedder | None = None
_lock = asyncio.Lock()


def build_embedder() -> BaseEmbedder:
    """按配置造 Embedding。改 .env 即可切 provider。"""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider in ("dashscope", "openai", "aliyun"):
        return DashScopeEmbedder()
    if provider in ("local", "sentence-transformers", "hf"):
        return LocalEmbedder()
    raise ValueError(f"未知的 EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


async def get_embedder() -> BaseEmbedder:
    """全局单例：模型只加载/连接一次，反复创建会浪费内存与连接。"""
    global _embedder
    if _embedder is None:
        async with _lock:
            if _embedder is None:
                _embedder = build_embedder()
                logger.info(f"Embedding 就绪: provider={settings.EMBEDDING_PROVIDER} model={_embedder.name}")
    return _embedder
