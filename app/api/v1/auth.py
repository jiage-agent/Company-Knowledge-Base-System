"""
认证接口。

登录为什么要传 tenant（公司名）而不是只传用户名（面试会问）：
    多租户系统里用户名只在租户内唯一 —— A 公司的"zhangsan"和 B 公司的"zhangsan"
    是两个不同的人。所以登录必须带租户标识，否则无法唯一定位用户。
    这也是数据库上那条 UniqueConstraint("tenant_id","username") 存在的理由。
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.core.logging import get_logger
from app.core.security import create_access_token, verify_password
from app.dependencies import PermDep, SessionDep
from app.models import AuditLog, Tenant, User
from app.schemas import LoginRequest, TokenResponse, UserInfo

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/login", response_model=TokenResponse, summary="登录获取访问令牌")
async def login(payload: LoginRequest, session: SessionDep):
    tenant = (
        await session.execute(select(Tenant).where(Tenant.name == payload.tenant))
    ).scalars().first()
    if tenant is None or not tenant.is_active:
        raise HTTPException(status_code=401, detail=f"租户不存在: {payload.tenant}")

    stmt = select(User).where(
        User.tenant_id == tenant.id, User.username == payload.username
    )
    user = (await session.execute(stmt)).scalars().first()

    # 安全细节：租户不存在/用户不存在/密码错误统一返回同一句提示，
    # 避免攻击者通过错误信息的差异枚举出"哪些用户名是存在的"。
    if user is None or not user.is_active or not verify_password(payload.password, user.hashed_password):
        logger.warning(f"登录失败 tenant={payload.tenant} username={payload.username}")
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_access_token(
        user_id=user.id,
        tenant_id=tenant.id,
        username=user.username,
        role=user.role,
        department=user.department,
        security_level=user.security_level,
    )
    session.add(AuditLog(
        tenant_id=tenant.id, user_id=user.id, action="login",
        resource=f"user:{user.username}",
        detail={"role": user.role, "department": user.department,
                "security_level": user.security_level},
    ))
    await session.commit()
    logger.info(f"登录成功 tenant={tenant.name} user={user.username} role={user.role}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserInfo, summary="查看当前登录身份与权限")
async def me(perm: PermDep, session: SessionDep):
    """
    演示用接口：一眼看清"我是谁、什么角色、什么密级"。
    面试演示权限隔离时，先用它展示两个账号的差异，再提问同样的文档问题看结果不同。
    """
    user = await session.get(User, perm.user_id) if perm.user_id else None
    tenant = await session.get(Tenant, perm.tenant_id)
    return UserInfo(
        id=perm.user_id or 0,
        username=user.username if user else "-",
        display_name=user.display_name if user else None,
        role=perm.role,
        department=perm.department,
        security_level=perm.security_level,
        tenant_id=perm.tenant_id,
        tenant_name=tenant.name if tenant else None,
    )
