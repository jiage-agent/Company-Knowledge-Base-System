"""
统一日志。

设计要点（面试可讲）：
1. 日志里带 trace_id：一次提问会横跨「解析 -> 检索 -> 重排 -> 生成」多个模块，
   出问题时能把同一请求的所有日志串起来，这是排障的基本功。
2. 控制台 + 文件双写：演示时看控制台，复盘时看 logs/app.log。
3. 用 contextvar 存 trace_id，业务代码零侵入，不用层层传参。
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import BASE_DIR

# 当前请求的追踪 ID（contextvar 是协程安全的：每个异步任务有独立副本）
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)-28s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


class TraceIdFilter(logging.Filter):
    """把 contextvar 里的 trace_id 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    trace_filter = TraceIdFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(trace_filter)
    # Windows 控制台默认 GBK，中文日志会乱码 -> 强制 UTF-8
    try:
        console.stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

    file_handler = RotatingFileHandler(
        Path(log_dir) / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(trace_filter)

    root.addHandler(console)
    root.addHandler(file_handler)

    # 降噪：第三方库日志太多
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                  "openai", "chromadb.telemetry", "asyncio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
