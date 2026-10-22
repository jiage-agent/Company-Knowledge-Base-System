"""
数据库引擎与会话管理。

设计要点（面试可讲）：
1. 用 SQLAlchemy 2.0 异步引擎 + SQLModel，
   从 SQLite 升级到 PostgreSQL 只需要改 .env 里的 DATABASE_URL：
   sqlite+aiosqlite:///./data/sqlite/app.db  ->  postgresql+asyncpg://user:pwd@host/db
   业务代码一行不用动 —— 这是"先用 SQLite 验证、后续升级"能够成立的前提。
2. 依赖注入用 get_session()，保证每个请求一个会话、用完自动关闭，
   避免连接泄漏（SQLite 尤其容易踩）。
3. SQLite 特有参数：check_same_thread=False（异步要用）。
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import BASE_DIR, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# SQLite 需要额外连接参数；换成 PostgreSQL 时这个 dict 会是空的
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,                     # 想看 SQL 就改成 True，调试时很好用
    connect_args=_connect_args,
    pool_pre_ping=True,             # 取连接前先 ping，避免拿到失效连接（生产必备）
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,         # commit 后对象仍可读，方便接口层继续用
    autoflush=False,
)


async def init_db() -> None:
    """
    建表。

    说明：开发期用 create_all 快速起步；一旦进入多人协作/生产，
    就要换成 Alembic 迁移（因为生产环境不允许 drop 重建表）。
    这里同时 import 所有模型，保证 metadata 里注册了全部表。
    """
    from app import models  # noqa: F401  触发模型注册

    settings.ensure_dirs()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info(f"数据库就绪: {settings.DATABASE_URL}")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：每个请求一个数据库会话。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_db() -> None:
    await engine.dispose()
