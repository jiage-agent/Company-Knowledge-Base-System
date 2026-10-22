"""
领域模型（Domain Model）—— 整个系统的契约层。

为什么模型要这么设计（面试高频追问，逐条都有答案）：

1. 为什么 chunk 表要单独存在，而不是把整篇文档塞进向量库？
   -> 一篇文档切分成 N 个块，块才是检索的最小单位；父子块策略要求
      "用小块做召回、用大块喂给大模型"，所以必须能按 parent_id 反查。

2. 为什么每个 chunk 都冗余 tenant_id，而不是靠 document 关联查？
   -> 向量库（Chroma/Milvus）不支持 JOIN。把权限字段下推到向量库过滤，
      必须让每个向量自带 tenant_id / security_level / department，
      否则只能在应用层后置过滤 —— 那会导致候选集被掏空、召回率骤降。

3. 为什么 document 要 is_active / version 而不是直接删？
   -> 文档更新时，旧块要软删除（保留审计与回溯），新版本号 +1；
      用户提问时只检索 is_active=True 的块。硬删除会导致
      向量库与关系库不一致且无法恢复。

4. 为什么 embedding_model 要落库？
   -> 换 embedding 模型（512维 -> 1024维）后，旧向量与新向量不在同一空间，
      不可混用。落库才能在迁移时精准找出"需要重建的块"。

5. 为什么 jwt 里带 tenant_id，而不是每请求查库？
   -> 权限上下文（租户/角色/部门/密级）随 token 传递，检索时直接用于过滤，
      省一次数据库往返，这是延迟与安全的折中。
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from sqlalchemy import Column, DateTime, Index, Text, UniqueConstraint
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

# Pydantic v2 把 model_ 开头视为保留字段名；embedding_model 等是业务字段，显式允许
_ALLOW_MODEL_NS = {"protected_namespaces": ()}

# 时区感知的时间类型（面试小知识）：
#   时间一律存 UTC 且带时区，展示时再按用户时区转换。
#   如果存"本地时间但不带时区"，跨时区部署、夏令时切换、日志对时全都会出错，
#   而且 SQLModel 0.0.48+ 会直接拒绝写入不带时区的 datetime（我们实测踩到过）。
AwareDatetime = Annotated[datetime, Column(DateTime(timezone=True))]


def utcnow() -> datetime:
    """统一用带时区的 UTC 当前时间。"""
    return datetime.now(timezone.utc)


# ============================================================
# 枚举
# ============================================================
class DocStatus(str, Enum):
    """文档处理状态机：上传 -> 解析中 -> 已入库，任一环节失败进 FAILED。"""

    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocType(str, Enum):
    """文档业务类型。不同业务类型用不同切分策略（合同按条款、制度按章节）。"""

    POLICY = "policy"          # 制度/规范
    CONTRACT = "contract"      # 合同
    REPORT = "report"          # 报告
    MANUAL = "manual"          # 手册/说明书
    OTHER = "other"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


# ============================================================
# 租户与用户（多租户 + 权限的根）
# ============================================================
class Tenant(SQLModel, table=True):
    """租户 = 一家公司/一个隔离单位。所有业务数据都挂 tenant_id。"""

    __tablename__ = "tenants"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True, index=True)
    is_active: bool = Field(default=True)
    created_at: AwareDatetime = Field(default_factory=utcnow)


class User(SQLModel, table=True):
    """
    用户。

    security_level：密级（1=公开 2=内部 3=机密 4=绝密）。
    过滤规则是"用户密级 >= 文档密级"，用数值比较即可，不用枚举硬编码。
    department：部门，用于"本部门文档本部门可见"的场景。
    """

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "username", name="uq_user_tenant_username"),)

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    username: str = Field(max_length=64, index=True)
    hashed_password: str = Field(max_length=200)
    display_name: str | None = Field(default=None, max_length=64)

    role: str = Field(default="member", max_length=32)        # admin | member
    department: str | None = Field(default=None, max_length=64, index=True)
    security_level: int = Field(default=1, index=True)

    is_active: bool = Field(default=True)
    created_at: AwareDatetime = Field(default_factory=utcnow)


# ============================================================
# 文档与块（知识层）
# ============================================================
class Document(SQLModel, table=True):
    """
    文档主表：一份原始文件一条记录。

    file_hash 用于秒传/去重：相同内容的文件不重复解析入库。
    version 用于增量更新：同一逻辑文档的第 N 次上传。
    """

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_doc_tenant_status", "tenant_id", "status"),
        Index("ix_doc_tenant_hash", "tenant_id", "file_hash"),
    )

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)

    title: str = Field(max_length=255)
    filename: str = Field(max_length=255)
    file_path: str = Field(max_length=500)
    file_type: str = Field(max_length=16)                     # pdf | docx | xlsx | txt | md
    file_size: int = Field(default=0)
    file_hash: str = Field(max_length=64, index=True)         # sha256，用于去重与版本判定

    doc_type: str = Field(default=DocType.OTHER.value, max_length=16)
    status: str = Field(default=DocStatus.PENDING.value, max_length=16, index=True)
    version: int = Field(default=1)
    is_active: bool = Field(default=True)
    error_message: str | None = Field(default=None, sa_column=Column(Text))

    # ---- 权限（文档级 ACL）----
    department: str | None = Field(default=None, max_length=64, index=True)
    security_level: int = Field(default=1, index=True)
    allowed_roles: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # ---- 统计 ----
    chunk_count: int = Field(default=0)
    char_count: int = Field(default=0)
    parse_seconds: float = Field(default=0.0)
    index_seconds: float = Field(default=0.0)

    uploader_id: int | None = Field(default=None, foreign_key="users.id")
    created_at: AwareDatetime = Field(default_factory=utcnow, index=True)
    updated_at: AwareDatetime = Field(default_factory=utcnow)


class DocumentChunk(SQLModel, table=True):
    """
    文档块（检索的最小单位）。

    父子块设计：
      - is_parent=True  的块：完整章节，存全文，用于"喂给大模型"
      - is_parent=False 的块：子块，向量化后用于"召回"
      - 子块 parent_id 指向父块，命中子块后自动带出父块全文
    这样解决"切太碎丢上下文、切太大召回不准"的矛盾。

    注意：chunk 冗余了 tenant_id / department / security_level，
    因为这些字段要下推到向量库做过滤（向量库不能 JOIN）。
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_chunk_tenant_doc", "tenant_id", "document_id"),
        Index("ix_chunk_tenant_active", "tenant_id", "is_active"),
        Index("ix_chunk_parent", "parent_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    chunk_uid: str = Field(max_length=64, unique=True, index=True)   # 与向量库主键一致
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    parent_id: int | None = Field(default=None, foreign_key="document_chunks.id")

    content: str = Field(sa_column=Column(Text))
    # BM25 检索语料（中文按空格分词后的形式），冗余存储是为了避免每次检索都重新分词
    search_text: str = Field(default="", sa_column=Column(Text))
    chunk_index: int = Field(default=0)
    is_parent: bool = Field(default=False, index=True)
    is_active: bool = Field(default=True, index=True)

    # ---- 溯源信息（引用来源要精确到页）----
    page_no: int | None = Field(default=None)
    section: str | None = Field(default=None, max_length=255)      # 所属章节标题
    block_type: str = Field(default="text", max_length=16)         # text | table | title

    # ---- 权限快照（写入向量库的部分）----
    doc_version: int = Field(default=1)
    department: str | None = Field(default=None, max_length=64)
    security_level: int = Field(default=1)
    allowed_roles: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # ---- 向量相关 ----
    embedding_model: str = Field(default="", max_length=64)
    embedding_dim: int = Field(default=0)
    token_count: int = Field(default=0)

    created_at: AwareDatetime = Field(default_factory=utcnow)


# ============================================================
# 会话与消息（在线问答）
# ============================================================
class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    id: int | None = Field(default=None, primary_key=True)
    session_uid: str = Field(max_length=64, unique=True, index=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    title: str = Field(default="新会话", max_length=100)
    created_at: AwareDatetime = Field(default_factory=utcnow)
    updated_at: AwareDatetime = Field(default_factory=utcnow)


class ChatMessage(SQLModel, table=True):
    """
    对话消息。assistant 消息保存「可验证的引用」：
      citations = [{"doc_id":1,"chunk_uid":"...","title":"员工手册.pdf","page_no":12,"score":0.83}]
    这是"答案可追溯"的落地方式 —— 前端可点击跳转到原文页。
    """

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_msg_session_created", "session_id", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chat_sessions.id", index=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    role: str = Field(max_length=16)

    content: str = Field(sa_column=Column(Text))
    citations: list[dict] = Field(default_factory=list, sa_column=Column(JSON))

    # ---- Agent 运行轨迹（面试演示的亮点：能看到"它怎么想的"）----
    intent: str | None = Field(default=None, max_length=32)
    retrieval_rounds: int = Field(default=0)
    confidence: float = Field(default=0.0)
    latency_ms: int = Field(default=0)
    trace_id: str | None = Field(default=None, max_length=32)
    retrieved_count: int = Field(default=0)

    created_at: AwareDatetime = Field(default_factory=utcnow)


# ============================================================
# 审计日志（企业合规的关键，也是权限演示的证据）
# ============================================================
class AuditLog(SQLModel, table=True):
    """
    审计：谁、什么时候、问了什么、命中了哪些文档、是否发生越权拦截。

    企业场景里"能证明数据没被越权访问"本身就是交付物，
    这也是简历上"权限隔离"最有力的证据（可以查表给面试官看）。
    """

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    user_id: int | None = Field(default=None, index=True)
    action: str = Field(max_length=32, index=True)   # upload | delete | query | login
    resource: str = Field(default="", max_length=255)
    detail: dict = Field(default_factory=dict, sa_column=Column(JSON))
    trace_id: str | None = Field(default=None, max_length=32)
    created_at: AwareDatetime = Field(default_factory=utcnow, index=True)


# ============================================================
# 评测记录（消融实验的数据来源）
# ============================================================
class EvalRecord(SQLModel, table=True):
    """
    每次评测跑一条。mode 对应消融实验的三组配置：
      dense_only | dense_bm25 | dense_bm25_rerank  (可再加 agentic)
    有了这张表，"混合检索比纯向量提升多少"这句话才有数字支撑。
    """

    __tablename__ = "eval_records"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(max_length=64, index=True)
    mode: str = Field(max_length=32, index=True)
    question: str = Field(sa_column=Column(Text))
    gold_answer: str | None = Field(default=None, sa_column=Column(Text))
    predicted_answer: str | None = Field(default=None, sa_column=Column(Text))
    gold_chunk_uids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    retrieved_chunk_uids: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    recall_at_k: float = Field(default=0.0)
    mrr: float = Field(default=0.0)
    ndcg: float = Field(default=0.0)
    latency_ms: int = Field(default=0)

    created_at: AwareDatetime = Field(default_factory=utcnow)
