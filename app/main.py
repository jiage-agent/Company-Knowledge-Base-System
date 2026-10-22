"""
FastAPI 应用入口。

启动方式：
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000
    然后打开 http://127.0.0.1:8000/docs  （Swagger 交互文档，演示主入口）

这里的几个工程细节（面试可讲）：
1. lifespan 里做启动/关闭：启动时建目录、建表、预热 embedding 模型；
   关闭时释放连接。用 lifespan 而不是 @app.on_event（后者在新版 FastAPI 已废弃）。
2. trace_id 中间件：给每个请求分配追踪 ID 并写入响应头 X-Trace-Id，
   前端报错时把 header 一贴就能定位全链路日志 —— 排障效率的关键。
3. 全局异常处理：把未捕获异常统一转成 JSON 并记录 trace_id，
   而不是把 Python 堆栈直接吐给调用方（既有安全考虑，也更好用）。
4. CORS 默认放开：本项目的调用方可能是本地前端页面或 Postman，
   开发期放开是合理的；生产环境必须收紧成白名单（这行注释是有意留下的）。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router, health_router
from app.config import settings
from app.core.logging import get_logger, new_trace_id, setup_logging, trace_id_var
from app.db.session import dispose_db, init_db

setup_logging("DEBUG" if settings.DEBUG else "INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动与关闭钩子。"""
    logger.info("=" * 72)
    logger.info(f"启动 {settings.APP_NAME} | env={settings.APP_ENV}")
    settings.ensure_dirs()
    await init_db()

    # 启动时预热：embedding 与重排都做一次探测调用。
    # 好处是"启动慢一点、首个请求快一点"，而且配置错误在启动阶段就暴露，
    # 不会等到用户第一次提问才报错（fail fast 原则）。
    try:
        from app.rag.embedding import get_embedder

        emb = await get_embedder()
        probe = await emb.embed_query("启动自检")
        logger.info(f"Embedding 自检通过: {emb.name} dim={len(probe)}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Embedding 自检失败（服务仍会启动，检索时将报错）: {str(e)[:200]}")

    try:
        from app.vectorstore.base import get_vector_store

        store = await get_vector_store()
        logger.info(f"向量库自检通过: {store.name} 现有向量 {await store.count()}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"向量库自检失败: {str(e)[:200]}")

    logger.info(f"LLM: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
    logger.info(f"重排: provider={settings.RERANK_PROVIDER} enabled={settings.RERANK_ENABLED}")
    logger.info(f"接口文档: http://127.0.0.1:8000/docs")
    logger.info("=" * 72)

    yield

    await dispose_db()
    logger.info("服务已关闭")


app = FastAPI(
    title="企业知识库问答系统（Agentic RAG）",
    description=(
        "基于 **LangGraph 多节点编排 + 混合检索（BM25 + 向量 + RRF）+ Cross-Encoder 重排** "
        "的企业知识库问答系统。\n\n"
        "**核心能力**\n"
        "- 多格式解析：PDF / Word / Excel / CSV / TXT / Markdown\n"
        "- 父子块索引：小块召回、大块生成\n"
        "- 混合检索：稠密（向量）+ 稀疏（BM25）双路召回，RRF 融合，重排精排\n"
        "- Agentic 编排：查询理解 -> 检索规划 -> 混合检索 -> 生成 -> Critic 校验（不足则改写查询重检索）\n"
        "- 多租户 + 文档级权限：租户 / 部门 / 密级 / 角色白名单四元组，权限条件下推到向量库\n"
        "- 引用溯源：答案带可点击的引用（文档 + 页码 + 相关度）\n"
        "- 效果可量化：内置消融实验（仅向量 / +混合检索 / +重排）\n\n"
        "**快速开始**：先调 `POST /api/v1/auth/login` 获取 token，"
        "点右上角 Authorize 填入即可调试其余接口。"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "认证", "description": "登录获取 JWT，令牌内含租户/角色/部门/密级"},
        {"name": "文档管理", "description": "上传、解析入库、查看切分结果、删除"},
        {"name": "问答", "description": "Agentic RAG 问答（可切换朴素 RAG 做对比）"},
        {"name": "管理", "description": "统计、组件健康检查、消融实验"},
        {"name": "健康检查", "description": "存活与就绪探针"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境必须改成具体域名白名单
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """为每个请求注入 trace_id，并在响应头回传。"""
    tid = request.headers.get("X-Trace-Id") or new_trace_id()
    trace_id_var.set(tid)
    t0 = time.perf_counter()
    response = await call_next(request)
    cost = int((time.perf_counter() - t0) * 1000)
    response.headers["X-Trace-Id"] = tid
    response.headers["X-Process-Time-Ms"] = str(cost)
    if not request.url.path.startswith(("/docs", "/openapi", "/redoc")):
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({cost}ms)")
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    兜底异常处理。

    生产实践：给用户返回"可读的提示 + trace_id"，堆栈只进日志。
    这样用户能报出 trace_id、我们能精确检索日志，
    同时不把内部路径、依赖版本等敏感信息暴露出去。
    """
    tid = trace_id_var.get()
    logger.exception(f"未处理异常 path={request.url.path} trace_id={tid}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误，请把 trace_id 提供给管理员以便定位",
            "trace_id": tid,
        },
    )


app.include_router(health_router)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["健康检查"], summary="服务信息与接口导航")
async def root() -> dict:
    """根路径：给出接口导航，方便演示时快速找到入口。"""
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "登录": f"{settings.API_V1_PREFIX}/auth/login",
            "上传文档": f"{settings.API_V1_PREFIX}/documents/upload",
            "文档列表": f"{settings.API_V1_PREFIX}/documents",
            "提问": f"{settings.API_V1_PREFIX}/chat/ask",
            "统计": f"{settings.API_V1_PREFIX}/admin/stats",
            "组件健康": f"{settings.API_V1_PREFIX}/admin/health/components",
            "消融实验": f"{settings.API_V1_PREFIX}/admin/eval/run",
            "就绪探针": "/ready",
        },
    }
