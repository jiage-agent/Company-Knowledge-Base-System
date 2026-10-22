"""
文档接口：上传、列表、详情、删除。

上传接口的设计要点（面试可讲）：
1. 文件 + 元数据一起提交（multipart/form-data）：
   元数据用 Form 字段而不是 JSON body，因为 multipart 只能有一个 body，
   这是 FastAPI/HTTP 的约束，也是常见的踩坑点。
2. 权限元数据（部门/密级/角色白名单）在上传时就确定并写入每个 chunk ——
   只有入库时带上权限字段，检索时才能把权限条件下推到向量库。
   如果入库不带、检索时再补，就需要回填全部向量，代价极高。
3. 当前为同步入库：上传请求会等到解析+向量化完成才返回。
   演示直观（上传完立刻能问），但大文件会超时。
   生产应改为任务队列：立即返回 task_id，用 /documents/{id} 轮询状态。
   代码里已按阶段切分（parse/index），迁移到 worker 只需换调用位置。
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from app.core.logging import get_logger
from app.core.permissions import PermissionContext
from app.dependencies import PermDep, SessionDep
from app.models import Document, DocumentChunk
from app.rag.ingest import (
    create_document,
    delete_document,
    ingest_document,
    save_upload,
    sha256_of,
)
from app.rag.parser import SUPPORTED_SUFFIXES
from app.schemas import ChunkOut, DeleteResult, DocumentOut, DocumentPage

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["文档管理"])

MAX_UPLOAD_MB = 30


@router.post("/upload", response_model=DocumentOut, summary="上传文档并入库（解析→切分→向量化）")
async def upload_document(
    session: SessionDep,
    perm: PermDep,
    file: UploadFile = File(..., description=f"支持格式: {', '.join(SUPPORTED_SUFFIXES)}"),
    title: str = Form("", description="文档标题，留空用文件名"),
    doc_type: str = Form("other", description="policy | contract | report | manual | other"),
    department: str = Form("", description="归属部门，留空=全公司可见"),
    security_level: int = Form(1, ge=1, le=4, description="密级 1公开 2内部 3机密 4绝密"),
    allowed_roles: str = Form("", description="额外授权角色，逗号分隔，如 hr,finance"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的格式 {suffix}，当前支持: {', '.join(SUPPORTED_SUFFIXES)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB}MB 限制")

    saved_path = save_upload(content, file.filename or f"upload{suffix}")
    file_hash = sha256_of(saved_path)
    roles = [r.strip() for r in allowed_roles.split(",") if r.strip()]

    doc = await create_document(
        session,
        tenant_id=perm.tenant_id,
        user_id=perm.user_id or 0,
        title=title,
        filename=file.filename or saved_path.name,
        file_path=saved_path,
        file_hash=file_hash,
        doc_type=doc_type,
        department=department,
        security_level=security_level,
        allowed_roles=roles,
    )
    await session.refresh(doc)

    # 已入库过的相同内容直接返回（幂等），不重复消耗 embedding 额度
    from app.models import DocStatus

    if doc.status != DocStatus.INDEXED.value:
        stats = await ingest_document(session, doc.id)
        await session.refresh(doc)
        if not stats.get("ok"):
            raise HTTPException(
                status_code=422,
                detail=f"入库失败: {stats.get('error', '未知错误')}",
            )
    return DocumentOut(**doc.model_dump())


@router.get("", response_model=DocumentPage, summary="文档列表（按权限过滤，只能看到自己有权访问的）")
async def list_documents(
    session: SessionDep,
    perm: PermDep,
    status: str | None = Query(None, description="按状态过滤: indexed/failed/..."),
    doc_type: str | None = Query(None, description="按业务类型过滤"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    conds = [Document.tenant_id == perm.tenant_id, Document.is_active == True]  # noqa: E712
    if not perm.is_admin:
        conds.append(Document.security_level <= perm.security_level)
    if status:
        conds.append(Document.status == status)
    if doc_type:
        conds.append(Document.doc_type == doc_type)

    total = (await session.execute(
        select(func.count()).select_from(Document).where(*conds)
    )).scalar() or 0

    rows = (await session.execute(
        select(Document).where(*conds)
        .order_by(Document.created_at.desc()).limit(limit).offset(offset)
    )).scalars().all()

    # 列表接口也要过一遍权限判定：密级相同但部门不同的文档不应出现在列表里。
    # 只在 SQL 层过滤密级是不够的 —— 部门维度也要拦（这是很多项目的漏洞点）。
    visible = [
        d for d in rows
        if perm.can_access({
            "tenant_id": d.tenant_id, "security_level": d.security_level,
            "department": d.department or "", "allowed_roles": d.allowed_roles or [],
            "is_active": d.is_active,
        })
    ]
    return DocumentPage(total=total, items=[DocumentOut(**d.model_dump()) for d in visible])


@router.get("/{document_id}", response_model=DocumentOut, summary="文档详情")
async def get_document(
    document_id: int,
    session: SessionDep,
    perm: PermDep,
):
    doc = await session.get(Document, document_id)
    if doc is None or doc.tenant_id != perm.tenant_id:
        # 跨租户访问返回 404 而不是 403：不泄漏"该 id 存在"这一信息
        raise HTTPException(status_code=404, detail="文档不存在")
    if not perm.can_access({
        "tenant_id": doc.tenant_id, "security_level": doc.security_level,
        "department": doc.department or "", "allowed_roles": doc.allowed_roles or [],
        "is_active": doc.is_active,
    }):
        raise HTTPException(status_code=403, detail="无权访问该文档")
    return DocumentOut(**doc.model_dump())


@router.get("/{document_id}/chunks", response_model=list[ChunkOut], summary="查看文档切分结果（演示切分效果用）")
async def list_chunks(
    document_id: int,
    session: SessionDep,
    perm: PermDep,
    only_children: bool = Query(True, description="只看子块（检索单元），false 看父块（生成单元）"),
    limit: int = Query(20, ge=1, le=200),
):
    doc = await session.get(Document, document_id)
    if doc is None or doc.tenant_id != perm.tenant_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    stmt = (
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.tenant_id == perm.tenant_id,
            DocumentChunk.is_parent == (not only_children),
        )
        .order_by(DocumentChunk.chunk_index)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        ChunkOut(
            id=c.id, chunk_uid=c.chunk_uid, chunk_index=c.chunk_index,
            is_parent=c.is_parent, page_no=c.page_no, section=c.section,
            block_type=c.block_type, content=c.content,
        )
        for c in rows
    ]


@router.delete("/{document_id}", response_model=DeleteResult, summary="删除文档（软删除 + 清理向量）")
async def remove_document(
    document_id: int,
    session: SessionDep,
    perm: PermDep,
):
    doc = await session.get(Document, document_id)
    if doc is None or doc.tenant_id != perm.tenant_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not perm.is_admin and doc.uploader_id != perm.user_id:
        raise HTTPException(status_code=403, detail="只有管理员或上传者本人可以删除")
    deleted = await delete_document(session, document_id, perm.tenant_id)
    return DeleteResult(
        document_id=document_id, deleted_chunks=deleted,
        message="文档已软删除，向量已从检索范围移除",
    )
