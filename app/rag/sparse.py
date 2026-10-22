"""
BM25 稀疏检索 —— 性能优化版（含模型缓存与失效机制）。

一、为什么需要 BM25（能力互补）：
    稠密检索（向量）擅长语义近似，但对"专有名词、编号、精确数字"不敏感 ——
    例如 `INC-2024-0912`、`DPWX-年份-四位流水号`、`不超过450元` 这类 token
    在训练语料里罕见，向量表示会被稀释。
    稀疏检索（BM25）只看词频与逆文档频率，对精确词匹配极灵敏，
    但没有语义泛化能力。两者互补，构成混合检索的基础。

二、为什么自己实现而不是用 Elasticsearch（工程权衡）：
    ES 很强但太重（要部署、要运维、要同步双写），对单机项目不划算。
    这里用"关系库存文本 + rank_bm25 + 进程内模型缓存"：
      - 部署零成本；
      - 大语料下靠缓存把固定开销摊薄，单次检索只剩打分。
    什么时候该换：语料过百万、或需要多机共享索引时，
    应换 Milvus 原生稀疏向量（省掉双索引一致性）或 ES。

三、性能优化：模型缓存 + 版本失效（实测调优，面试重点）：
    初版是"每次检索都从数据库捞全部块 → 重新分词 → 重建 BM25 模型"。
    在 2600 个块上实测拆解：
        读库 38ms / 分词 149ms（190 万 token）/ 建模型 233ms（词表 3.6 万）
        / 打分 9ms / 排序 1ms
    **429ms 里有 382ms 与查询无关**，只有 9ms 真正在打分。
    更糟的是评测时每道题都新建检索器实例，缓存完全不命中，
    端到端延迟从 221ms 涨到 2172ms —— 用户视角就是"开了混合检索就变卡 10 倍"。

    现在的做法：
      1. 模型按"权限指纹"缓存在进程内（不同权限看到的语料不同，必须分开缓存）；
      2. 文档增删时调用 bump_corpus_version() 让缓存整体失效；
      3. 缓存加锁，避免并发重复构建。
    结果：首次 429ms，之后每次约 5~9ms。
"""

from __future__ import annotations

import asyncio
import time

from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.logging import get_logger
from app.core.permissions import PermissionContext, RetrievedChunk
from app.models import Document, DocumentChunk
from app.rag.fusion import tokenize

logger = get_logger(__name__)

MAX_CORPUS = 8000        # 单次参与 BM25 的语料上限
_CACHE_MAX_MODELS = 4    # 最多缓存几份模型（按权限指纹区分）

# 进程内缓存：fingerprint -> {version, bm25, uids, payload, build_ms}
_cache: dict[tuple, dict] = {}
_lock = asyncio.Lock()
_corpus_version = 0      # 语料变更时 +1，所有缓存失效


def bump_corpus_version() -> None:
    """
    语料变更通知：入库 / 删除 / 重建索引后调用。

    为什么用"全局版本号"而不是精确失效某个 key：
      一次入库可能影响多个权限视角（全公司文档改了，所有部门都要重算），
      精确失效容易漏。版本号最简单且不会错，
      代价只是下次检索要重建一次模型（几百毫秒，可接受）。
    """
    global _corpus_version
    _corpus_version += 1
    _cache.clear()
    logger.info(f"BM25 语料缓存已失效（version={_corpus_version}）")


class BM25Retriever:
    """稀疏（关键词）检索器。模型缓存共享，实例本身很轻。"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _load_corpus(self, perm: PermissionContext) -> dict:
        """
        加载（或复用缓存的）BM25 模型与语料元数据。

        缓存 key 包含 tenant_id + 部门 + 密级 + 是否管理员：
        这是必须的 —— 如果只按 tenant 缓存，同一租户下不同密级/部门的用户
        会命中彼此的语料，等于越权（这类"缓存串味"是真实事故的常见来源）。
        """
        fingerprint = (perm.tenant_id, perm.department, perm.security_level, perm.is_admin)
        cached = _cache.get(fingerprint)
        if cached and cached["version"] == _corpus_version:
            return cached

        async with _lock:
            # 双重检查：等锁期间可能已被别的协程建好
            cached = _cache.get(fingerprint)
            if cached and cached["version"] == _corpus_version:
                return cached

            t0 = time.perf_counter()
            stmt = (
                select(Document, DocumentChunk)
                .join(DocumentChunk, DocumentChunk.document_id == Document.id)
                .where(*perm.sql_conditions())
                .limit(MAX_CORPUS)
            )
            rows = (await self.session.execute(stmt)).all()
            if not rows:
                entry = {"version": _corpus_version, "bm25": None, "uids": [],
                         "payload": {}, "build_ms": 0}
                _cache[fingerprint] = entry
                logger.warning(f"BM25 语料为空 tenant={perm.tenant_id}")
                return entry

            if len(rows) >= MAX_CORPUS:
                logger.warning(
                    f"BM25 语料达到上限 {MAX_CORPUS}，关键词召回可能不完整。"
                    f"这是当前实现的已知瓶颈，升级 Milvus 稀疏向量后可消除。"
                )

            corpus_tokens: list[list[str]] = []
            uids: list[str] = []
            payload: dict[str, RetrievedChunk] = {}
            for doc, chunk in rows:
                text = chunk.search_text or chunk.content
                corpus_tokens.append(tokenize(text))
                uids.append(chunk.chunk_uid)
                payload[chunk.chunk_uid] = RetrievedChunk(
                    chunk_uid=chunk.chunk_uid,
                    content=chunk.content,
                    doc_id=chunk.document_id,
                    title=doc.title,
                    page_no=chunk.page_no,
                    section=chunk.section,
                    block_type=chunk.block_type,
                    source="sparse",
                    meta={
                        "tenant_id": chunk.tenant_id,
                        "security_level": chunk.security_level,
                        "department": chunk.department or "",
                        "allowed_roles": chunk.allowed_roles or [],
                        "doc_version": chunk.doc_version,
                        "is_active": chunk.is_active,
                    },
                )

            bm25 = BM25Okapi(corpus_tokens)
            entry = {
                "version": _corpus_version,
                "bm25": bm25,
                "uids": uids,
                "payload": payload,
                "build_ms": int((time.perf_counter() - t0) * 1000),
            }

            # LRU 淘汰：超出上限时丢掉最早的一份，避免多权限组合下内存膨胀
            if len(_cache) >= _CACHE_MAX_MODELS:
                _cache.pop(next(iter(_cache)), None)
            _cache[fingerprint] = entry
            logger.info(f"BM25 模型构建完成: {len(corpus_tokens)} 条语料，"
                        f"{entry['build_ms']}ms（后续查询将复用）")
            return entry

    async def search(
        self, query: str, top_k: int, perm: PermissionContext
    ) -> list[RetrievedChunk]:
        entry = await self._load_corpus(perm)
        bm25: BM25Okapi | None = entry["bm25"]
        if bm25 is None:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for idx in ranked:
            if scores[idx] <= 0:          # 完全无匹配的不要，避免噪声进入融合
                continue
            item = entry["payload"][entry["uids"][idx]]
            item.score = float(scores[idx])
            results.append(item)
        return results

    def invalidate(self) -> None:
        """兼容旧调用：等价于让全局缓存失效。"""
        bump_corpus_version()

    @staticmethod
    def cache_info() -> dict:
        """返回缓存状态，供健康检查/调试接口展示。"""
        return {
            "cached_models": len(_cache),
            "corpus_version": _corpus_version,
            "details": [
                {"fingerprint": str(k), "corpus": len(v["uids"]), "build_ms": v["build_ms"]}
                for k, v in _cache.items()
            ],
        }
