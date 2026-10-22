# ============================================================
# 生产镜像（多阶段构建）
#
# 为什么用多阶段：
#   构建阶段需要 uv 和编译工具链，运行阶段只需要解释器 + 依赖，
#   分阶段能把镜像体积从约 1.5GB 压到 400MB 左右，也更少攻击面。
#
# 说明：当前演示环境用本地 Python 直跑（Windows 上 Docker 需要 WSL2），
#       这个 Dockerfile 用于 Linux 部署与容器化演示。
# ============================================================

FROM python:3.13-slim AS builder

# uv：比 pip 快一个数量级的依赖解析器（官方静态二进制）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# 先只拷依赖清单，让依赖层可以被 Docker 缓存复用
#（改业务代码不会导致重新装依赖，构建从几分钟降到几秒）
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 再拷源码并安装项目本身
COPY app ./app
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

# 安全实践：不用 root 跑服务（容器逃逸后的影响面更小）
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    TZ=Asia/Shanghai

COPY --from=builder --chown=appuser:appuser /app /app

# 数据目录挂载点（SQLite / Chroma / 上传文件都落在这里，便于持久化）
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app/data /app/logs
VOLUME ["/app/data"]

USER appuser
EXPOSE 8000

# 健康检查：用 /ready（真实检查数据库与向量库），而不是 /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
