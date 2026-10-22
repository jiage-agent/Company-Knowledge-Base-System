"""
管理接口：统计、组件健康检查、评测实验。

为什么把评测做成接口而不仅是脚本（面试可讲）：
    "效果可回归"是产品级系统的基本要求。
    做成接口后，它可以被 CI 调用（每次改分块策略/换模型都跑一遍），
    而不是靠开发者本地记得手动跑 —— 把质量保证变成流程，而不是靠自觉。
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from app.config import settings
from app.core.logging import get_logger
from app.core.permissions import PermissionContext
from app.dependencies import AdminDep, PermDep, SessionDep
from app.eval.harness import format_report, run_eval
from app.models import Document, DocumentChunk, Tenant, User
from app.schemas import ComponentOut, EvalRequest, EvalResponse, ModeMetric, StatsOut
from app.vectorstore.base import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["管理"])


@router.get("/stats", response_model=StatsOut, summary="系统统计（文档/块/向量/租户/用户）")
async def stats(session: SessionDep, perm: AdminDep):
    async def count(model, *conds) -> int:
        return (await session.execute(
            select(func.count()).select_from(model).where(*conds)
        )).scalar() or 0

    store = await get_vector_store()
    return StatsOut(
        tenant_count=await count(Tenant),
        user_count=await count(User),
        document_count=await count(Document, Document.is_active == True),  # noqa: E712
        chunk_count=await count(DocumentChunk, DocumentChunk.is_active == True),  # noqa: E712
        vector_count=await store.count(),
        indexed_document_count=await count(Document, Document.status == "indexed"),
    )


@router.get("/health/components", response_model=list[ComponentOut],
            summary="各依赖组件健康检查（LLM / Embedding / 重排 / 向量库 / 数据库）")
async def component_health(session: SessionDep, perm: AdminDep):
    """
    分组件的健康检查。

    为什么这个接口在演示时很加分：
      面试官常问"你这个系统依赖这么多外部服务，挂了怎么办"。
      打开这个页面，逐项显示每个依赖的状态和延迟，
      再说明"向量库挂了会怎样、重排挂了会降级"，
      这比口头解释有说服力得多。
    """
    results: list[ComponentOut] = []

    # 1. 数据库
    t0 = time.perf_counter()
    try:
        await session.execute(select(func.count()).select_from(Tenant))
        results.append(ComponentOut(
            name="database (SQLite)", ok=True,
            detail=settings.DATABASE_URL.split("///")[-1],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        ))
    except Exception as e:  # noqa: BLE001
        results.append(ComponentOut(name="database", ok=False, detail=str(e)[:150]))

    # 2. 向量库
    t0 = time.perf_counter()
    try:
        store = await get_vector_store()
        n = await store.count()
        results.append(ComponentOut(
            name=f"vector store ({store.name})", ok=True, detail=f"{n} 条向量",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        ))
    except Exception as e:  # noqa: BLE001
        results.append(ComponentOut(name="vector store", ok=False, detail=str(e)[:150]))

    # 3. Embedding
    t0 = time.perf_counter()
    try:
        from app.rag.embedding import get_embedder
        emb = await get_embedder()
        vec = await emb.embed_query("健康检查")
        results.append(ComponentOut(
            name=f"embedding ({emb.name})", ok=len(vec) > 0,
            detail=f"provider={settings.EMBEDDING_PROVIDER} dim={len(vec)}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        ))
    except Exception as e:  # noqa: BLE001
        results.append(ComponentOut(name="embedding", ok=False, detail=str(e)[:150]))

    # 4. 重排
    t0 = time.perf_counter()
    try:
        from app.rag.reranker import build_reranker
        rr = build_reranker()
        # 注意：不同 provider 用的模型名不同（api 用 RERANK_MODEL，local 用 RERANK_LOCAL_MODEL），
        # 之前这里写死成 RERANK_MODEL，导致本地模式下健康检查显示的模型名是错的
        model_name = (settings.RERANK_LOCAL_MODEL if settings.RERANK_PROVIDER.lower() == "local"
                      else settings.RERANK_MODEL)
        device = getattr(rr, "_device", "")
        results.append(ComponentOut(
            name=f"reranker ({rr.name})", ok=settings.RERANK_ENABLED,
            detail=f"provider={settings.RERANK_PROVIDER} model={model_name}"
                   + (f" device={device}" if device else "")
                   + ("" if settings.RERANK_ENABLED else " (disabled)"),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        ))
    except Exception as e:  # noqa: BLE001
        results.append(ComponentOut(name="reranker", ok=False, detail=str(e)[:150]))

    # 5. LLM（真实调用一次，会花几厘钱，但能确认链路真的通）
    from app.rag.llm import check_llm
    ok, detail, ms = await check_llm()
    results.append(ComponentOut(name=f"llm ({settings.LLM_MODEL})", ok=ok, detail=detail, latency_ms=ms))

    # 6. BM25 模型缓存状态
    # 为什么把它暴露出来：排查"混合检索突然变慢"或"新文档搜不到"时，
    # 第一件事就是看缓存有没有命中 / 语料版本有没有刷新。
    try:
        from app.rag.sparse import BM25Retriever

        info = BM25Retriever.cache_info()
        results.append(ComponentOut(
            name="bm25 sparse index", ok=True,
            detail=f"已缓存 {info['cached_models']} 份模型 / 语料版本 {info['corpus_version']} / "
                   f"明细 {info['details']}",
            latency_ms=0,
        ))
    except Exception as e:  # noqa: BLE001
        results.append(ComponentOut(name="bm25 sparse index", ok=False, detail=str(e)[:150]))

    return results


@router.post("/eval/run", response_model=EvalResponse,
             summary="运行消融实验（对比 仅向量 / +混合检索 / +重排 三组配置）")
async def eval_run(
    payload: EvalRequest,
    session: SessionDep,
    perm: AdminDep,
):
    """
    跑评测。

    注意 limit 参数：全量评测会调用很多次 embedding（每道题都要向量化查询），
    演示时建议先 limit=6 跑通，再决定要不要全量跑 —— 这既是成本意识，也避免现场超时。
    """
    result = await run_eval(
        session,
        tenant_id=perm.tenant_id,
        modes=payload.modes,
        top_k=payload.top_k,
        limit=payload.limit,
    )
    logger.info("评测完成:\n" + format_report(result))
    return EvalResponse(
        run_id=result["run_id"],
        dataset_size=result["dataset_size"],
        metrics=[ModeMetric(**m) for m in result["metrics"]],
        detail=result["detail"][:100],
    )
