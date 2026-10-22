"""
权限上下文与过滤规则 —— 企业知识库最核心的安全边界。

本项目的权限模型是「四元组」，而不是只有密级：

    tenant_id       租户隔离：A 公司的数据 B 公司永远查不到（硬隔离）
    department      部门：文档有归属部门，用户只能看本部门 + 全公司公开的
    security_level  密级：1公开 2内部 3机密 4绝密；规则是"用户级别 >= 文档级别"
    allowed_roles   白名单角色：某些文档额外授权给指定角色（如 hr、finance）

为什么不能只在关系库里判权限、必须在向量库也带权限字段（面试必考）：
    向量检索是"先算相似度、后返回 Top-K"。
    如果先算出 Top-K 再在应用层过滤掉无权限的，会出现两个问题：
      1. 越权泄漏风险：过滤前的候选里已经包含了机密文档内容，
         只要有一处日志/异常打印了候选，就等于泄漏；
      2. 召回塌缩：Top-10 里 8 条被过滤掉，只剩 2 条给大模型，
         答案质量骤降（这就是"后置过滤"的致命伤）。
    正确做法：把权限条件「下推」到向量库（Chroma 的 where / Milvus 的 expr），
    让无权限的数据在检索阶段就不进入候选集。

    另外还要保留一次应用层兜底校验（defense in depth）：
    库的过滤语法写错时，兜底能拦住越权，这是"不信任单点"的工程习惯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import User


@dataclass
class PermissionContext:
    """一次检索的权限上下文，由 JWT 解析而来（不额外查库）。"""

    tenant_id: int
    user_id: int | None = None
    role: str = "member"
    department: str | None = None
    security_level: int = 1
    is_admin: bool = False

    @classmethod
    def from_user(cls, user: User) -> "PermissionContext":
        return cls(
            tenant_id=user.tenant_id,
            user_id=user.id,
            role=user.role or "member",
            department=user.department,
            security_level=user.security_level or 1,
            is_admin=(user.role == "admin"),
        )

    @classmethod
    def from_token_payload(cls, payload: dict[str, Any]) -> "PermissionContext":
        """从 JWT payload 构造（检索时不再查库，省一次往返）。"""
        role = payload.get("role") or "member"
        return cls(
            tenant_id=int(payload.get("tenant_id") or 0),
            user_id=int(payload["sub"]) if payload.get("sub") else None,
            role=role,
            department=payload.get("department"),
            security_level=int(payload.get("security_level") or 1),
            is_admin=(role == "admin"),
        )

    # ---------------- 过滤条件构造 ----------------
    def chroma_where(self) -> dict[str, Any]:
        """
        构造 Chroma 的 where 过滤条件。

        Chroma 语法要点：
          - 多条件并列默认是 AND，但显式写 $and 更清晰、兼容性更好；
          - $in / $lte 是操作符；$ne 用于"非本部门"的公开文档。
        注意：allowed_roles 的"角色交集"判断放在应用层兜底做，
        因为 Chroma 对数组字段没有交集操作符（Milvus 有 array_contains，可下推）。
        """
        conds: list[dict[str, Any]] = [
            {"tenant_id": {"$eq": self.tenant_id}},
            {"is_active": {"$eq": True}},
        ]
        if not self.is_admin:
            # 密级：用户能看的最高密级
            conds.append({"security_level": {"$lte": self.security_level}})
            # 部门：本部门 或 全公司公开（department 为空）
            if self.department:
                conds.append(
                    {"$or": [
                        {"department": {"$eq": self.department}},
                        {"department": {"$eq": ""}},
                    ]}
                )
        return {"$and": conds} if len(conds) > 1 else conds[0]

    def sql_conditions(self) -> list:
        """
        构造 SQLAlchemy 的过滤条件（BM25 关键词检索走 SQLite，同样要下推权限）。

        为什么要两套：稠密检索走向量库、稀疏检索走关系库，
        两条路都必须带权限条件，漏掉任何一条就是一个越权漏洞。
        """
        from sqlmodel import col

        from app.models import DocumentChunk

        conds = [
            col(DocumentChunk.tenant_id) == self.tenant_id,
            col(DocumentChunk.is_active) == True,  # noqa: E712
            col(DocumentChunk.is_parent) == False,  # noqa: E712  只召回子块
        ]
        if not self.is_admin:
            conds.append(col(DocumentChunk.security_level) <= self.security_level)
            if self.department:
                conds.append(
                    (col(DocumentChunk.department) == self.department)
                    | (col(DocumentChunk.department) == "")
                    | (col(DocumentChunk.department).is_(None))
                )
        return conds

    def can_access(self, meta: dict[str, Any]) -> bool:
        """
        应用层兜底校验（defense in depth）。
        即使向量库过滤条件写错，这一步也能拦住越权数据。
        """
        if int(meta.get("tenant_id", -1)) != self.tenant_id:
            return False
        if not meta.get("is_active", True):
            return False
        if self.is_admin:
            return True
        if int(meta.get("security_level", 99)) > self.security_level:
            return False
        # 角色白名单：文档声明了 allowed_roles 时，用户角色必须在其中（或为空=不限）
        allowed = meta.get("allowed_roles") or []
        if isinstance(allowed, str):
            allowed = [r for r in allowed.split(",") if r]
        if allowed and self.role not in allowed:
            return False
        dept = meta.get("department") or ""
        if dept and self.department and dept != self.department:
            return False
        return True


@dataclass
class RetrievedChunk:
    """统一检索结果结构：稠密/稀疏/融合/重排各阶段都用它，避免结构转换带来的 bug。"""

    chunk_uid: str
    content: str
    score: float = 0.0
    doc_id: int = 0
    title: str = ""
    page_no: int | None = None
    section: str | None = None
    parent_uid: str | None = None
    parent_content: str | None = None
    block_type: str = "text"
    source: str = ""            # dense | sparse | fused | rerank —— 便于调试与评测
    meta: dict[str, Any] = field(default_factory=dict)

    def to_citation(self, index: int) -> dict[str, Any]:
        """转成引用结构，接口层直接返回给前端做角标跳转。"""
        return {
            "index": index,
            "chunk_uid": self.chunk_uid,
            "doc_id": self.doc_id,
            "title": self.title or self.source,
            "page_no": self.page_no,
            "section": self.section,
            "score": round(self.score, 4),
            "snippet": self.content[:180],
        }
