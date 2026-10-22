"""
健康检查与就绪探针。

为什么要区分 /health 与 /ready（面试可讲）：
    liveness（存活）：进程还在跑吗？挂了就重启容器。
    readiness（就绪）：依赖都通吗？没通就不要把流量导进来。
    混在一起会导致"依赖没起来 -> 探针失败 -> 容器被反复重启"的雪崩。
    单机项目看起来无所谓，但这套约定是上线部署的常识。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.core.logging import get_logger
from app.schemas import HealthOut
from app.vectorstore.base import get_vector_store

logger = get_logger(__name__)
router = APIRouter(tags=["健康检查"])


@router.get("/health", summary="存活探针（不检查依赖，永远快速返回）")
async def health() -> dict:
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}


@router.get("/ready", response_model=HealthOut, summary="就绪探针（真实检查数据库与向量库）")
async def ready() -> HealthOut:
    database = "unknown"
    vector = "unknown"
    try:
        from sqlalchemy import text

        from app.db.session import engine

        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        database = "ok"
    except Exception as e:  # noqa: BLE001
        database = f"error: {str(e)[:80]}"

    try:
        store = await get_vector_store()
        n = await store.count()
        vector = f"ok ({n} vectors)"
    except Exception as e:  # noqa: BLE001
        vector = f"error: {str(e)[:80]}"

    overall = "ok" if database == "ok" and vector.startswith("ok") else "degraded"
    return HealthOut(
        status=overall,
        app=settings.APP_NAME,
        env=settings.APP_ENV,
        database=database,
        vector_store=vector,
        llm_model=f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
        embedding_model=f"{settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}",
        rerank_enabled=settings.RERANK_ENABLED,
    )
