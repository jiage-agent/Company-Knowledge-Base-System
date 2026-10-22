"""
依赖注入：从请求里解析出「当前用户 + 权限上下文」。

设计要点（面试可讲）：
1. 认证与授权分开：authenticate 只负责"你是谁"（解析 JWT），
   authorize 负责"你能干什么"（角色/密级判断）。混在一起会导致权限逻辑散落各处。
2. 权限上下文（PermissionContext）是从 token 直接构造的，不查库 ——
   每次请求少一次数据库往返；代价是权限变更后需要重新登录（换取的是延迟）。
   如果要求"改密级立即生效"，就得改成每请求查库或用缓存 + 版本号失效，
   这是一个有意识的权衡，面试时可以主动说明。
3. get_current_user 用 HTTPBearer 而不是 OAuth2PasswordBearer：
   Swagger 里直接点 Authorize 粘贴 token 即可调试，演示更顺手。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger, new_trace_id, trace_id_var
from app.core.permissions import PermissionContext
from app.core.security import decode_access_token
from app.db.session import get_session

logger = get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False, description="登录后把 access_token 填到这里")


async def get_trace_id() -> str:
    """为每个请求生成 trace_id，贯穿日志、审计与响应头。"""
    tid = trace_id_var.get()
    if tid == "-":
        tid = new_trace_id()
        trace_id_var.set(tid)
    return tid


async def get_current_perm(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> PermissionContext:
    """解析 JWT 得到权限上下文；失败返回 401。"""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供访问令牌，请先调用 /api/v1/auth/login 获取 token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    perm = PermissionContext.from_token_payload(payload)
    if perm.tenant_id <= 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌缺少租户信息")
    return perm


async def require_admin(perm: PermissionContext = Depends(get_current_perm)) -> PermissionContext:
    """管理接口的守卫：只有 admin 角色能访问。"""
    if not perm.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"需要管理员权限，当前角色: {perm.role}",
        )
    return perm


# 便捷别名：用 Annotated 而不是把 Depends 当默认值传给参数
# （写成 perm: PermissionContext = PermDep 会报 "Depends is not callable"，
#   这是 FastAPI 新版依赖注入必须用 Annotated 的典型坑）
SessionDep = Annotated[AsyncSession, Depends(get_session)]
PermDep = Annotated[PermissionContext, Depends(get_current_perm)]
AdminDep = Annotated[PermissionContext, Depends(require_admin)]
TraceDep = Annotated[str, Depends(get_trace_id)]
