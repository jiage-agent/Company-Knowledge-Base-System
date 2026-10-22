"""
Milvus 实现（生产升级路径）。

当前状态：**未启用**（.env 里 VECTOR_STORE_TYPE=chroma）。
保留这个文件是为了让"升级路径"看得见、摸得着，而不是一句空话。

为什么升级到 Milvus（三条硬理由，面试可直接讲）：
    1. 过滤能力：Milvus 支持标量字段表达式过滤（expr）和数组字段的 array_contains，
       权限条件可以完整下推，不需要像 Chroma 那样把角色交集判断挪到应用层兜底。
    2. 稀疏向量：Milvus 2.4+ 原生支持 sparse vector（SPLADE/BM25 风格），
       可以把稠密+稀疏放在同一个 collection 做混合检索，
       省掉现在"向量库 + 自建 BM25 索引"的双索引一致性问题。
    3. 规模与并发：分布式架构、多副本、分片（partition），
       支持按 tenant_id 做 partition key，多租户下既能隔离又能水平扩展。

迁移需要改的东西（这就是当初做抽象层的回报）：
    - 新增本文件的完整实现（add / search / delete_by_document / count）
    - .env 改一行：VECTOR_STORE_TYPE=milvus，并在 docker-compose 里起 milvus
    - 上层检索、Agent、评测代码：一行都不用改

如何补齐实现（留给后续迭代的清晰步骤）：
    pip install pymilvus
    from pymilvus import MilvusClient
    client = MilvusClient(uri="http://localhost:19530")
    # 1. create_collection：字段 id(VARCHAR,主键) / vector(FLOAT_VECTOR,dim) /
    #    tenant_id(INT64) / department(VARCHAR) / security_level(INT64) /
    #    allowed_roles(ARRAY<VARCHAR>) / is_active(BOOL) / content(VARCHAR)
    # 2. create_index：metric_type="COSINE", index_type="HNSW"
    # 3. search：expr 用 f'tenant_id == {t} and security_level <= {lv} ...' 下推权限
    # 4. 写完后跑 scripts/eval.py 对比 Chroma 与 Milvus 的 Recall@K / P95 延迟，
    #    把对比数据写进 README —— 这就是"升级带来的收益"的量化证据。
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.permissions import PermissionContext, RetrievedChunk
from app.vectorstore.base import BaseVectorStore, VectorRecord

logger = get_logger(__name__)


class MilvusStore(BaseVectorStore):
    name = "milvus"

    def __init__(self) -> None:
        raise NotImplementedError(
            "Milvus 适配器尚未实现。当前版本使用 ChromaDB 做开发验证。\n"
            "升级步骤见本文件顶部注释（约 200 行代码，上层无需改动）。"
        )

    async def add(self, records: list[VectorRecord]) -> int:  # pragma: no cover
        raise NotImplementedError

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        perm: PermissionContext,
        where_extra: dict | None = None,
    ) -> list[RetrievedChunk]:  # pragma: no cover
        raise NotImplementedError

    async def delete_by_document(self, document_id: int) -> int:  # pragma: no cover
        raise NotImplementedError

    async def count(self, tenant_id: int | None = None) -> int:  # pragma: no cover
        raise NotImplementedError

    async def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError
