"""
全局配置中心。

设计要点（面试可讲）：
1. 所有外部模型（LLM / Embedding / Reranker / 向量库）全部通过环境变量注入，
   业务代码只依赖抽象接口，不依赖具体厂商 —— 这是"热切换"能成立的前提。
2. 用 pydantic-settings 而不是散落的 os.getenv：好处是启动时就会校验类型、
   缺配置立刻报错（fail fast），而不是等跑到一半才崩。
3. 配置项按"领域"分组，而不是按"谁用"分组，便于换模型时只改一段。
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/config.py 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- App ----------------
    APP_NAME: str = "Company-Knowledge-Base-System"
    APP_ENV: str = "dev"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # ---------------- LLM（生成模型） ----------------
    LLM_PROVIDER: str = "deepseek"
    LLM_MODEL: str = "deepseek-chat"
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 90

    # ---------------- Embedding（向量模型） ----------------
    # provider: dashscope(阿里云百炼, 走 OpenAI 兼容协议) | local(本地 sentence-transformers)
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "bge-m3:latest"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 10
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_LOCAL_MODEL: str = "BAAI/bge-small-zh-v1.5"
    HF_ENDPOINT: str = "https://hf-mirror.com"
    # Ollama 本地服务（provider=ollama 时使用）
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_EMBED_MODEL: str = "bge-m3:latest"

    # ---------------- Reranker（重排） ----------------
    RERANK_ENABLED: bool = True
    RERANK_PROVIDER: str = "api"          # api | local | none
    RERANK_MODEL: str = "gte-rerank-v2"   # api 模式下的模型名
    RERANK_LOCAL_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_TOP_N: int = 30
    RERANK_DEVICE: str = "auto"

    # ---------------- 向量库 ----------------
    VECTOR_STORE_TYPE: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "enterprise_kb"

    # ---------------- 关系库 ----------------
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/sqlite/app.db"

    # ---------------- 切分 ----------------
    CHUNK_STRATEGY: str = "recursive"
    CHUNK_SIZE: int = 240
    CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 900
    SEMANTIC_THRESHOLD: float = 0.62

    # ---------------- 检索 ----------------
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    FUSION_TOP_K: int = 20
    RERANK_TOP_K: int = 5
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    # 相关性下限：分阶段独立标定（用数据标出来的，不是拍脑袋）
    #
    # 标定方法（scripts/calibrate 实验）：对评测集每题取 Top-1 分数，
    # 按"期望作答 / 期望拒答"两组统计分布，找能把两类分开的阈值：
    #
    #   阶段        无答案题最高  有答案题最低  最佳阈值  分开准确率
    #   稠密检索      0.4505       0.3948      0.340     98.7%
    #   RRF 融合      0.0164       0.0161      ——        90.7%（完全重叠，不可分）
    #   重排          0.1143       0.1592      0.115     100%
    #
    # 结论（面试可讲）：
    #   RRF 分数是 1/(k+rank) 的累加，本质是"相对名次"，**不携带绝对相关性信息**，
    #   所以不能用它判断"库里到底有没有答案"——这正是必须接一个重排阶段的硬理由：
    #   重排给出的是经过模型判定的绝对相关性，才能支撑"拒答"这个决策。
    MIN_RELEVANCE_SCORE_DENSE: float = 0.34
    MIN_RELEVANCE_SCORE_RRF: float = 0.0      # 相对名次，绝对阈值无意义（见上表）
    MIN_RELEVANCE_SCORE_RERANK: float = 0.115
    # 兼容旧配置项：显式设置它会覆盖上面三项
    MIN_RELEVANCE_SCORE: float = 0.0

    # ---------------- Agent 编排 ----------------
    MAX_RETRY: int = 2
    GRAPH_RECURSION_LIMIT: int = 25
    AGENT_TIMEOUT_SECONDS: int = 90

    # ---------------- 可观测 ----------------
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "company-kb-agentic-rag"

    # ---------------- 安全 ----------------
    JWT_SECRET_KEY: str = "dev-only-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # ---------------- 派生路径（自动建目录） ----------------
    @computed_field
    @property
    def SQLITE_PATH(self) -> str:
        """从 DATABASE_URL 反解出 sqlite 文件路径，启动时自动建目录。"""
        url = self.DATABASE_URL
        if "sqlite" in url and ":///" in url:
            return url.split(":///")[-1]
        return ""

    def ensure_dirs(self) -> None:
        """启动时把需要的目录一次性建好，避免运行中因为目录不存在而报错。"""
        for d in [
            BASE_DIR / "data",
            BASE_DIR / "data" / "sqlite",
            BASE_DIR / "data" / "uploads",
            BASE_DIR / "data" / "chroma",
            BASE_DIR / "logs",
        ]:
            d.mkdir(parents=True, exist_ok=True)
        sqlite_path = self.SQLITE_PATH
        if sqlite_path:
            (BASE_DIR / sqlite_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """单例：全进程只解析一次 .env。"""
    return Settings()


settings = get_settings()
