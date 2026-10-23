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


@router.get("/warm", summary="预热接口（演示前调用，避免首个提问吃冷启动）")
async def warm() -> dict:
    """
    主动预热：把向量化、重排两条链路的模型都跑一遍再返回。

    为什么需要这个接口（演示时很实用）：
      服务刚启动时模型可能还在加载（本地 Cross-Encoder 首载约 8 秒，
      Ollama 首字节也要预热）。如果这期间恰好有用户提问，
      重排/向量化会超时，用户看到的是"未找到可靠依据"这种假性拒答 ——
      很难解释，而且看起来像系统能力不行。

      演示前先 curl 一次 /warm，等它返回 ok 再开始提问，
      就能保证每一次提问都是"热"的。

    返回各环节耗时，也能顺手当成"性能自检"。
    """
    import time

    result: dict = {"embedding": None, "rerank": None, "retrieval": None}
    t0 = time.perf_counter()

    # 1) 向量化预热
    try:
        from app.rag.embedding import get_embedder

        emb = await get_embedder()
        t = time.perf_counter()
        vec = await emb.embed_query("预热")
        result["embedding"] = {"ok": True, "dim": len(vec),
                               "ms": int((time.perf_counter() - t) * 1000)}
    except Exception as e:  # noqa: BLE001
        result["embedding"] = {"ok": False, "error": str(e)[:150]}

    # 2) 重排预热（本地模型要加载权重到显存）
    try:
        from app.rag.reranker import build_reranker

        rr = build_reranker()
        t = time.perf_counter()
        if hasattr(rr, "_ensure_model"):
            import asyncio as _asyncio

            await _asyncio.to_thread(rr._ensure_model)
        result["rerank"] = {"ok": True, "name": rr.name,
                            "device": getattr(rr, "_device", ""),
                            "ms": int((time.perf_counter() - t) * 1000)}
    except Exception as e:  # noqa: BLE001
        result["rerank"] = {"ok": False, "error": str(e)[:150]}

    # 3) 完整检索链路跑一次（会触发 BM25 语料加载）
    try:
        from sqlmodel import select

        from app.core.permissions import PermissionContext
        from app.db.session import AsyncSessionLocal
        from app.models import Tenant
        from app.rag.hybrid import HybridRetriever

        async with AsyncSessionLocal() as session:
            tenant = (await session.execute(select(Tenant).order_by(Tenant.id))).scalars().first()
            if tenant is not None:
                perm = PermissionContext(tenant_id=tenant.id, role="admin", security_level=4,
                                         is_admin=True)
                t = time.perf_counter()
                r = HybridRetriever(session)
                chunks, _dbg = await r.retrieve("预热", perm, top_k=3)
                result["retrieval"] = {"ok": True, "chunks": len(chunks),
                                       "ms": int((time.perf_counter() - t) * 1000)}
    except Exception as e:  # noqa: BLE001
        result["retrieval"] = {"ok": False, "error": str(e)[:150]}

    result["total_ms"] = int((time.perf_counter() - t0) * 1000)
    result["ready"] = all(
        (result.get(k) or {}).get("ok", False) for k in ("embedding", "rerank", "retrieval")
    )
    return result
