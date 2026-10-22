"""
入库流水线：文件 -> 解析 -> 切分 -> 向量化 -> 双写（关系库 + 向量库）。

为什么不写成"一个函数干到底"（面试可讲）：
    入库是个长流程（大文件解析+向量化可能几十秒），任何一步都可能失败。
    如果失败只抛异常，用户看到的是"上传失败"，但数据库里可能已经留下半截数据。
    所以这里用「状态机 + 分段记录」：
        PENDING -> PARSING -> INDEXING -> INDEXED / FAILED
    每一步的耗时和错误都落在 document 表上，
    出问题能精确定位到是哪一步（这是"可运维"和"能跑"的区别）。

幂等设计：
    上传前先算文件 sha256，同一租户下相同内容直接返回已有文档，
    不重复解析、不重复向量化 —— 既省成本又避免向量库里出现重复数据。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.core.logging import get_logger, new_trace_id, trace_id_var
from app.core.permissions import PermissionContext
from app.models import AuditLog, DocStatus, Document, DocumentChunk, utcnow
from app.rag.chunking import build_chunks
from app.rag.embedding import get_embedder
from app.rag.fusion import tokenize
from app.rag.parser import parse_document
from app.vectorstore.base import VectorRecord, get_vector_store

logger = get_logger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_upload(content: bytes, filename: str) -> Path:
    """
    落盘。

    为什么加 uuid 前缀：不同用户可能上传同名文件（"员工手册.pdf"），
    直接按原名保存会互相覆盖 —— 这是多租户系统里非常容易忽略的坑。
    保留原名片段是为了排障时还能看出是什么文件。
    """
    settings.ensure_dirs()
    from app.config import BASE_DIR

    upload_dir = BASE_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    target.write_bytes(content)
    return target


async def create_document(
    session: AsyncSession,
    *,
    tenant_id: int,
    user_id: int,
    title: str,
    filename: str,
    file_path: Path,
    file_hash: str,
    doc_type: str,
    department: str,
    security_level: int,
    allowed_roles: list[str],
) -> Document:
    """创建文档记录（状态 PENDING），并处理"同内容已存在"的情况。"""
    stmt = select(Document).where(
        Document.tenant_id == tenant_id,
        Document.file_hash == file_hash,
        Document.is_active == True,  # noqa: E712
    )
    existing = (await session.execute(stmt)).scalars().first()
    if existing:
        logger.info(f"文件内容已存在（hash 相同），跳过重复入库: doc_id={existing.id}")
        return existing

    doc = Document(
        tenant_id=tenant_id,
        title=title or Path(filename).stem,
        filename=filename,
        file_path=str(file_path),
        file_type=Path(filename).suffix.lower().lstrip("."),
        file_size=file_path.stat().st_size,
        file_hash=file_hash,
        doc_type=doc_type,
        status=DocStatus.PENDING.value,
        department=department,
        security_level=security_level,
        allowed_roles=allowed_roles,
        uploader_id=user_id,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def ingest_document(session: AsyncSession, document_id: int) -> dict:
    """
    执行入库。返回统计信息（成功与否都返回，便于接口层展示）。

    当前为同步执行（请求内完成）。生产环境应改为任务队列（Celery/arq）异步执行：
    接口立即返回 task_id，前端轮询状态。
    这里保持同步是为了演示"一条链路看得清楚"，代码里已把流程切成独立函数，
    拆到 worker 里只需换调用方式，不需要重写逻辑。
    """
    trace_id_var.set(new_trace_id())
    t_start = time.perf_counter()

    doc = await session.get(Document, document_id)
    if doc is None:
        return {"ok": False, "error": f"文档不存在: {document_id}"}

    stats: dict = {"document_id": doc.id, "ok": False}

    try:
        # ---------- 第一步：解析 ----------
        doc.status = DocStatus.PARSING.value
        doc.updated_at = utcnow()
        await session.commit()

        t_parse = time.perf_counter()
        blocks = parse_document(Path(doc.file_path))
        doc.parse_seconds = round(time.perf_counter() - t_parse, 3)
        doc.char_count = sum(len(b.text) for b in blocks)
        stats["blocks"] = len(blocks)
        stats["chars"] = doc.char_count
        logger.info(f"解析完成 doc_id={doc.id} blocks={len(blocks)} chars={doc.char_count}")

        # ---------- 第二步：切分（父块 + 子块）----------
        doc.status = DocStatus.INDEXING.value
        await session.commit()

        chunk_result = await build_chunks(blocks, document_id=doc.id, version=doc.version)
        stats.update(chunk_result.stats)
        if not chunk_result.children:
            raise ValueError("切分后没有得到任何子块，文档内容可能为空")

        # ---------- 第三步：向量化 ----------
        t_index = time.perf_counter()
        embedder = await get_embedder()
        children = chunk_result.children
        texts = [c.content for c in children]
        vectors = await embedder.embed_documents(texts)
        if len(vectors) != len(children):
            raise RuntimeError(f"向量数量与块数量不一致: {len(vectors)} vs {len(children)}")

        # ---------- 第四步：写关系库（先写库，再写向量库）----------
        # 顺序为什么重要：关系库是"事实来源"（source of truth）。
        # 若先写向量库成功、写库失败，会出现"检索得到但查不到元数据/权限"的幽灵数据；
        # 反过来只是"暂时检索不到"，可以靠重跑补偿 —— 宁可少，不可脏。
        id_to_uid: dict[int, str] = {}
        parent_objs: list[DocumentChunk] = []
        for p in chunk_result.parents:
            uid = f"d{doc.id}v{doc.version}p{p.parent_index}"
            obj = DocumentChunk(
                chunk_uid=uid,
                tenant_id=doc.tenant_id,
                document_id=doc.id,
                content=p.content,
                search_text=" ".join(tokenize(p.content)),
                chunk_index=p.parent_index,
                is_parent=True,
                page_no=p.page_no,
                section=p.section,
                block_type=p.block_type,
                doc_version=doc.version,
                department=doc.department or "",
                security_level=doc.security_level,
                allowed_roles=doc.allowed_roles or [],
                embedding_model=embedder.name,
                embedding_dim=0,          # 父块不向量化
            )
            session.add(obj)
            parent_objs.append(obj)
        await session.flush()             # 拿到父块自增 id
        for p, obj in zip(chunk_result.parents, parent_objs):
            id_to_uid[p.parent_index] = obj.chunk_uid

        child_objs: list[DocumentChunk] = []
        for c in children:
            obj = DocumentChunk(
                chunk_uid=c.chunk_uid,
                tenant_id=doc.tenant_id,
                document_id=doc.id,
                parent_id=None,
                content=c.content,
                search_text=" ".join(tokenize(c.content)),
                chunk_index=c.chunk_index,
                is_parent=False,
                page_no=c.page_no,
                section=c.section,
                block_type=c.block_type,
                doc_version=doc.version,
                department=doc.department or "",
                security_level=doc.security_level,
                allowed_roles=doc.allowed_roles or [],
                embedding_model=embedder.name,
                embedding_dim=len(vectors[0]) if vectors else 0,
                token_count=c.token_count,
            )
            session.add(obj)
            child_objs.append(obj)
        await session.flush()

        # 回填子块 -> 父块的引用
        for obj, c in zip(child_objs, children):
            obj.parent_id = next(
                (p.id for p in parent_objs if p.chunk_index == c.parent_index), None
            )

        # ---------- 第五步：写向量库 ----------
        store = await get_vector_store()
        records = [
            VectorRecord(
                chunk_uid=obj.chunk_uid,
                embedding=vec,
                content=obj.content,
                metadata={
                    "doc_id": doc.id,
                    "tenant_id": doc.tenant_id,
                    "title": doc.title,
                    "page_no": obj.page_no,
                    "section": obj.section or "",
                    "block_type": obj.block_type,
                    "parent_uid": id_to_uid.get(children[i].parent_index, ""),
                    "doc_version": doc.version,
                    "department": doc.department or "",
                    "security_level": doc.security_level,
                    "allowed_roles": doc.allowed_roles or [],
                    "is_active": True,
                },
            )
            for i, (obj, vec) in enumerate(zip(child_objs, vectors))
        ]
        await store.add(records)

        # ---------- 第六步：更新文档状态 ----------
        doc.chunk_count = len(child_objs)
        doc.index_seconds = round(time.perf_counter() - t_index, 3)
        doc.status = DocStatus.INDEXED.value
        doc.error_message = None
        doc.updated_at = utcnow()

        session.add(AuditLog(
            tenant_id=doc.tenant_id, user_id=doc.uploader_id, action="upload",
            resource=f"document:{doc.id}",
            detail={"title": doc.title, "chunks": len(child_objs),
                    "vectors": len(records), "security_level": doc.security_level},
            trace_id=trace_id_var.get(),
        ))
        await session.commit()

        stats["vectors"] = len(records)
        stats["ok"] = True
        stats["elapsed_s"] = round(time.perf_counter() - t_start, 2)
        logger.info(
            f"入库完成 doc_id={doc.id} 块={len(child_objs)} 向量={len(records)} "
            f"解析={doc.parse_seconds}s 索引={doc.index_seconds}s 总={stats['elapsed_s']}s"
        )

        # 语料变了 -> 让 BM25 模型缓存失效，否则新入库的文档在关键词检索里"看不见"
        # （这类"缓存没失效导致新数据搜不到"是很隐蔽的线上事故）
        from app.rag.sparse import bump_corpus_version

        bump_corpus_version()

    except Exception as e:  # noqa: BLE001
        await session.rollback()
        doc = await session.get(Document, document_id)
        if doc:
            doc.status = DocStatus.FAILED.value
            doc.error_message = str(e)[:800]
            doc.updated_at = utcnow()
            await session.commit()
        logger.exception(f"入库失败 doc_id={document_id}: {e}")
        stats["error"] = str(e)

    return stats


async def delete_document(session: AsyncSession, document_id: int, tenant_id: int) -> int:
    """
    删除文档（软删除 + 清理向量）。

    为什么要软删除而不是物理删除（面试高频）：
      1. 审计与回溯：企业场景要能回答"3 个月前那份文档写了什么"；
      2. 向量库重建有成本：软删除只需把 is_active 置 False，
         一旦误删可以快速恢复（物理删除要重新解析+向量化整篇文档）；
      3. 数据一致性：软删除是幂等的，重跑不会出错。
    但向量必须真的从检索范围里去掉（is_active=False + 向量库元数据同步），
    否则"已删除文档仍能被检索到"就是数据泄漏。
    """
    doc = await session.get(Document, document_id)
    if doc is None or doc.tenant_id != tenant_id:
        return 0

    stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    chunks = (await session.execute(stmt)).scalars().all()
    for c in chunks:
        c.is_active = False

    doc.is_active = False
    doc.status = DocStatus.FAILED.value
    doc.updated_at = utcnow()

    store = await get_vector_store()
    deleted = await store.delete_by_document(document_id)

    session.add(AuditLog(
        tenant_id=tenant_id, user_id=None, action="delete",
        resource=f"document:{document_id}",
        detail={"chunks": len(chunks), "vectors": deleted},
        trace_id=trace_id_var.get(),
    ))
    await session.commit()

    # 删除同样要让 BM25 缓存失效，否则已删内容还能被关键词检索到（等于数据泄漏）
    from app.rag.sparse import bump_corpus_version

    bump_corpus_version()
    logger.info(f"文档已删除 doc_id={document_id} 软删块={len(chunks)} 清向量={deleted}")
    return deleted
