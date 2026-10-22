"""
ChromaDB 实现（开发验证阶段）。

选它做开发期的原因（面试可讲）：
    零运维、pip 装完就能用、本地持久化，适合单机开发与演示。
它的短板正好是升级 Milvus 的理由：
    1. 元数据过滤能力弱（没有数组交集、表达式能力有限），权限过滤要拆一部分到应用层；
    2. 单机文件存储，没有分布式/副本，扛不住多用户并发与大数据量；
    3. 没有原生稀疏向量支持，BM25 只能自己维护（本项目就是自己实现的）。
把这三条讲清楚，"为什么升级 Milvus"就不是跟风，而是有依据的技术决策。

实现细节：
    - 用 embedding_function=None 自己算向量：向量化逻辑统一走我们的 Embedding 抽象层，
      这样同一份向量可以喂给 Chroma 或 Milvus，不被某个库的默认模型绑死。
    - 距离用 cosine，相似度 = 1 - 距离（cosine distance ∈ [0,2]，同向为 0）。
    - 写入用 upsert（不是 add）：chunk_uid 不变时覆盖，保证重建索引幂等。
"""

from __future__ import annotations

import asyncio
import os

from app.config import settings
from app.core.logging import get_logger
from app.core.permissions import PermissionContext, RetrievedChunk
from app.vectorstore.base import BaseVectorStore, VectorRecord

logger = get_logger(__name__)


class ChromaStore(BaseVectorStore):
    name = "chroma"

    def __init__(self) -> None:
        # 关掉匿名遥测，避免启动时联网卡顿（国内网络环境常见坑）
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        from app.config import BASE_DIR

        self._persist_dir = str((BASE_DIR / settings.CHROMA_PERSIST_DIR.replace("./", "")).resolve())
        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=None,          # 向量由我们自己算
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Chroma 就绪: dir={self._persist_dir} collection={settings.CHROMA_COLLECTION} "
                    f"现有向量={self._collection.count()}")

    # ---------------- 写入 ----------------
    async def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0

        def _do() -> int:
            self._collection.upsert(
                ids=[r.chunk_uid for r in records],
                embeddings=[r.embedding for r in records],
                documents=[r.content for r in records],
                metadatas=[_sanitize_meta(r.metadata) for r in records],
            )
            return len(records)

        n = await asyncio.to_thread(_do)
        logger.info(f"写入向量 {n} 条")
        return n

    # ---------------- 检索 ----------------
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        perm: PermissionContext,
        where_extra: dict | None = None,
    ) -> list[RetrievedChunk]:
        where = _merge_where(perm.chroma_where(), where_extra)

        def _do():
            return self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        raw = await asyncio.to_thread(_do)
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for uid, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            # 兜底权限校验：即使 where 条件写错，也不能把无权限内容交出去
            if not perm.can_access(meta):
                logger.warning(f"权限兜底拦截 chunk={uid} "
                               f"(tenant={meta.get('tenant_id')} level={meta.get('security_level')} "
                               f"dept={meta.get('department')})")
                continue
            results.append(
                RetrievedChunk(
                    chunk_uid=uid,
                    content=doc or "",
                    score=max(0.0, 1.0 - float(dist)),      # cosine distance -> 相似度
                    doc_id=int(meta.get("doc_id", 0)),
                    title=meta.get("title", ""),
                    page_no=meta.get("page_no"),
                    section=meta.get("section"),
                    parent_uid=meta.get("parent_uid"),
                    block_type=meta.get("block_type", "text"),
                    source="dense",
                    meta=meta,
                )
            )
        return results

    # ---------------- 删除 ----------------
    async def delete_by_document(self, document_id: int) -> int:
        def _do() -> int:
            existing = self._collection.get(where={"doc_id": {"$eq": int(document_id)}}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)

        n = await asyncio.to_thread(_do)
        logger.info(f"按文档删除向量 doc_id={document_id} -> {n} 条")
        return n

    async def delete_by_uids(self, uids: list[str]) -> int:
        if not uids:
            return 0

        def _do() -> int:
            self._collection.delete(ids=list(uids))
            return len(uids)

        return await asyncio.to_thread(_do)

    # ---------------- 统计 ----------------
    async def count(self, tenant_id: int | None = None) -> int:
        def _do() -> int:
            if tenant_id is None:
                return self._collection.count()
            got = self._collection.get(where={"tenant_id": {"$eq": int(tenant_id)}}, include=[])
            return len(got.get("ids") or [])

        return await asyncio.to_thread(_do)

    async def reset(self) -> None:
        def _do():
            self._client.delete_collection(settings.CHROMA_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )

        await asyncio.to_thread(_do)
        logger.warning("向量集合已清空")


# ============================================================
# 辅助函数
# ============================================================
def _sanitize_meta(meta: dict) -> dict:
    """
    Chroma 的 metadata 只接受 str / int / float / bool，
    列表要序列化成字符串（allowed_roles 就是这样处理的），None 必须去掉。
    这类"库的类型限制"如果不处理，会在写入时才报错，属于典型踩坑点。
    """
    out: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ",".join(str(x) for x in v)
        elif isinstance(v, dict):
            import json

            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _merge_where(perm_where: dict, extra: dict | None) -> dict:
    """把权限条件与业务条件（如限定文档类型）合并成 AND。"""
    if not extra:
        return perm_where
    return {"$and": [perm_where, extra]}
