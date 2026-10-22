"""
API 契约层（对外接口的输入输出模型）。

为什么单独一层：接口的"形状"是前后端/调用方之间的合同。
把它和数据库模型分开，好处是：
  1. 数据库字段（如 hashed_password）永远不会意外泄漏到接口返回里；
  2. 接口可以按需裁剪/组合字段，不被表结构绑死。
这是"内部模型不外泄"的基本功，也是面试可讲的点。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)


# ============================================================
# 认证
# ============================================================
class LoginRequest(_Base):
    tenant: str = Field(description="租户（公司）标识", examples=["acme"])
    username: str = Field(examples=["zhangsan"])
    password: str = Field(examples=["pass1234"])


class TokenResponse(_Base):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="有效期（秒）")


class UserInfo(_Base):
    id: int
    username: str
    display_name: str | None = None
    role: str
    department: str | None = None
    security_level: int
    tenant_id: int
    tenant_name: str | None = None


# ============================================================
# 文档
# ============================================================
class DocumentUploadMeta(_Base):
    """
    上传文档时随文件一起提交的元数据。

    description 写清楚是有意为之：Swagger 页面直接能看到这些字段怎么填，
    演示时不用解释，面试官自己就能点着测。
    """

    title: str = Field(default="", description="文档标题，留空则用文件名")
    doc_type: str = Field(default="other", description="policy | contract | report | manual | other")
    department: str = Field(default="", description="归属部门，留空=全公司可见")
    security_level: int = Field(default=1, ge=1, le=4, description="密级 1公开 2内部 3机密 4绝密")
    allowed_roles: str = Field(default="", description="额外允许的角色，逗号分隔，如 admin,hr")


class DocumentOut(_Base):
    id: int
    title: str
    filename: str
    file_type: str
    file_size: int
    doc_type: str
    status: str
    version: int
    is_active: bool
    department: str | None
    security_level: int
    chunk_count: int
    char_count: int
    parse_seconds: float
    index_seconds: float
    error_message: str | None
    created_at: datetime


class DocumentPage(_Base):
    total: int
    items: list[DocumentOut]


class ChunkOut(_Base):
    id: int
    chunk_uid: str
    chunk_index: int
    is_parent: bool
    page_no: int | None
    section: str | None
    block_type: str
    content: str


class DeleteResult(_Base):
    document_id: int
    deleted_chunks: int
    message: str


# ============================================================
# 问答
# ============================================================
class AskRequest(_Base):
    question: str = Field(min_length=1, examples=["公司年假有多少天？"])
    session_uid: str | None = Field(default=None, description="会话ID，不传则自动新建（多轮对话用）")
    top_k: int = Field(default=5, ge=1, le=20)
    use_agent: bool = Field(default=True, description="true=走 Agentic RAG 多节点编排；false=走朴素 RAG 单链")
    doc_type: str | None = Field(default=None, description="限定只在某类文档里检索，如 policy")


class Citation(_Base):
    """引用来源：精确到文档 + 页码 + 相关度，前端可点击定位原文。"""

    index: int = Field(description="正文里的角标序号，如 [1]")
    chunk_uid: str
    doc_id: int
    title: str
    page_no: int | None = None
    section: str | None = None
    score: float = 0.0
    snippet: str = ""


class AskResponse(_Base):
    session_uid: str
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, description="置信度 0~1")
    refused: bool = Field(default=False, description="是否因证据不足拒答（而非编造）")

    # ---- Agent 运行轨迹：演示时能讲清"它内部做了什么决策" ----
    intent: str = ""
    need_retrieval: bool = True
    planned_strategy: str = ""
    rewritten_queries: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 0
    retrieved_count: int = 0
    critic_passed: bool = False
    critic_reason: str = ""
    latency_ms: int = 0
    trace_id: str = ""


class MessageOut(_Base):
    id: int
    role: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    intent: str | None = None
    confidence: float = 0.0
    latency_ms: int = 0
    created_at: datetime


class SessionOut(_Base):
    session_uid: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


# ============================================================
# 管理 / 运维
# ============================================================
class StatsOut(_Base):
    tenant_count: int
    user_count: int
    document_count: int
    chunk_count: int
    vector_count: int
    indexed_document_count: int


class HealthOut(_Base):
    status: str
    app: str
    env: str
    database: str
    vector_store: str
    llm_model: str
    embedding_model: str
    rerank_enabled: bool


class ComponentOut(_Base):
    name: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0


class EvalRequest(_Base):
    modes: list[str] = Field(default_factory=lambda: ["dense_only", "dense_bm25", "dense_bm25_rerank"])
    top_k: int = Field(default=5, ge=1, le=20)
    limit: int = Field(default=0, description="只跑前 N 条，0=全量")
    use_llm_answer: bool = Field(default=False, description="是否生成答案（费 token，只算检索指标时可关）")


class ModeMetric(_Base):
    mode: str
    recall_at_k: float
    mrr: float
    ndcg: float
    hit_rate: float
    avg_latency_ms: float
    questions: int


class EvalResponse(_Base):
    run_id: str
    metrics: list[ModeMetric]
    dataset_size: int
    detail: list[dict[str, Any]] = Field(default_factory=list)
