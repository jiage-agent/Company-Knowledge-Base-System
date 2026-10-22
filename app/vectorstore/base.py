"""
向量库抽象层 —— 「Chroma 先用、后续换 Milvus」能成立的关键。

为什么必须抽象（面试可讲，也是简历上"迁移成本低"的真实证据）：
    如果检索代码里到处是 chroma_collection.query(...)，
    换 Milvus 就要改遍入库、检索、删除、统计所有地方，风险极高。
    把差异收敛到一个接口里之后：
      - 换库 = 新增一个实现类 + 改 .env 一行；
      - 上层（混合检索、Agent 节点、评测）完全不用动；
      - 还能同时保留两套实现，用于做性能对比（消融实验的素材）。

接口设计要点：
    1. search() 直接接收 PermissionContext —— 强制"权限下推"成为接口契约的一部分。
       这样任何人写新实现都不可能忘记带权限条件（把安全做成默认，而不是靠自觉）。
    2. 返回统一的 RetrievedChunk，避免各库返回结构不同导致上层写分支。
    3. 所有方法 async：Milvus/HTTP 客户端天然异步，Chroma 本地客户端是同步的，
       在实现里用线程池包一层，保持上层调用方式一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.permissions import PermissionContext, RetrievedChunk
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorRecord:
    """写入向量库的一条记录（= 一个子块）。"""

    __slots__ = ("chunk_uid", "embedding", "content", "metadata")

    def __init__(self, chunk_uid: str, embedding: list[float], content: str, metadata: dict) -> None:
        self.chunk_uid = chunk_uid
        self.embedding = embedding
        self.content = content
        self.metadata = metadata


class BaseVectorStore(ABC):
    """向量库统一接口。"""

    name: str = "base"

    @abstractmethod
    async def add(self, records: list[VectorRecord]) -> int:
        """批量写入/覆盖（chunk_uid 相同则覆盖，保证幂等）。"""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        perm: PermissionContext,
        where_extra: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        向量检索。

        where_extra：额外的业务过滤条件（例如"只在制度类文档里检索"），
        与权限条件做 AND 合并后一起下推到库。
        """

    @abstractmethod
    async def delete_by_document(self, document_id: int) -> int:
        """按文档删除全部向量（文档更新/删除时调用）。"""

    @abstractmethod
    async def count(self, tenant_id: int | None = None) -> int:
        """统计向量数量（健康检查/统计接口用）。"""

    @abstractmethod
    async def reset(self) -> None:
        """清空（仅用于重建索引/评测）。"""

    async def close(self) -> None:
        return None


# ============================================================
# 工厂 + 单例
# ============================================================
_store: BaseVectorStore | None = None


def build_vector_store() -> BaseVectorStore:
    """
    按配置构造向量库。

    VECTOR_STORE_TYPE=chroma  -> ChromaStore（开发验证）
    VECTOR_STORE_TYPE=milvus  -> MilvusStore（生产升级，见 milvus_store.py）
    """
    from app.config import settings

    kind = settings.VECTOR_STORE_TYPE.lower()
    if kind == "chroma":
        from app.vectorstore.chroma_store import ChromaStore

        return ChromaStore()
    if kind == "milvus":
        from app.vectorstore.milvus_store import MilvusStore

        return MilvusStore()
    raise ValueError(f"未知的 VECTOR_STORE_TYPE: {settings.VECTOR_STORE_TYPE}")


async def get_vector_store() -> BaseVectorStore:
    """全局单例：向量库客户端只初始化一次。"""
    global _store
    if _store is None:
        _store = build_vector_store()
        logger.info(f"向量库就绪: {_store.name}")
    return _store
