"""
安全模块：密码哈希 + JWT 签发校验。

设计要点（面试可讲）：
1. 密码绝不存明文，用 bcrypt 加盐哈希（同一密码两次哈希结果不同），不可逆。
2. JWT 里放 tenant_id / role / department / security_level ——
   这样每次请求不用查库就知道"你是谁、属于哪个租户、能看什么密级"，
   检索时的权限过滤直接用 token 里的信息，省掉一次数据库往返。
3. 直接用 bcrypt 库而不是 passlib：passlib 1.7.4 与新版 bcrypt 存在兼容性告警，
   少一层封装少一个坑（这是真实项目里常见的踩坑）。
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------- 密码 ----------------
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # 哈希串格式非法直接判失败，不让异常冒到接口层（避免 500）
        return False


# ---------------- JWT ----------------
def create_access_token(
    user_id: int,
    tenant_id: int,
    username: str,
    role: str = "member",
    department: str | None = None,
    security_level: int = 1,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "tenant_id": tenant_id,
        "username": username,
        "role": role,
        "department": department,
        "security_level": security_level,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """解析 token；过期或签名错误都返回 None，由调用方决定返回 401。"""
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.warning("token 已过期")
        return None
    except jwt.PyJWTError as e:
        logger.warning(f"token 解析失败: {e}")
        return None
