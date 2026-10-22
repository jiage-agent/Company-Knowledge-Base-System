"""切换 embedding 模型后的完整重建脚本。

为什么必须"全量重建"而不是"增量补"（面试可以讲）：
    不同的 embedding 模型把同一段文本映射到**完全不同的向量空间**，
    即使维度相同（text-embedding-v4 和 bge-m3 都是 1024 维）也不能混用：
      - 旧向量是 A 模型算的，新 query 是 B 模型算的，
        两者算余弦相似度毫无意义（相当于拿中文词典查英文单词）；
      - 正确做法是清空向量库，用新模型把所有块重新向量化。
    这也是为什么 chunk 表要记录 embedding_model 字段 ——
    真实生产里要能查出"哪些块的向量是旧模型算的"，精准重建。

本脚本做的事：
    1. 清空向量库集合（Chroma）
    2. 删除所有 document_chunks（父块+子块）
    3. 把 documents 状态重置为 PENDING（保留文档元数据与权限，避免重新上传）
    4. 用新 embedding 模型逐个重建索引
    5. 输出核对：向量数 / 块数 / 模型名
"""
import asyncio
import sys
import time

sys.path.insert(0, r"D:\develop\Company-Knowledge-Base-System")

from sqlmodel import select

from app.core.logging import setup_logging
from app.db.session import AsyncSessionLocal, init_db
from app.models import DocStatus, Document, DocumentChunk
from app.rag.embedding import get_embedder
from app.rag.ingest import ingest_document
from app.rag.sparse import bump_corpus_version
from app.vectorstore.base import get_vector_store


async def main() -> None:
    setup_logging("WARNING")
    await init_db()

    emb = await get_embedder()
    print("=" * 80)
    print(f"重建目标 Embedding: provider={emb.name} dim={emb.dim}")
    print("=" * 80)

    store = await get_vector_store()

    # ---------- 1) 清空向量库 ----------
    await store.reset()
    print(f"[1] 向量库已清空，当前向量数={await store.count()}")

    # ---------- 2) 清空块表 + 重置文档状态 ----------
    async with AsyncSessionLocal() as session:
        docs = (await session.execute(
            select(Document).where(Document.is_active == True)  # noqa: E712
        )).scalars().all()
        targets = [{"id": d.id, "title": d.title, "path": d.file_path} for d in docs]

        chunks = (await session.execute(select(DocumentChunk))).scalars().all()
        for c in chunks:
            await session.delete(c)
        for d in docs:
            d.status = DocStatus.PENDING.value
            d.chunk_count = 0
            d.version = d.version or 1
        await session.commit()
        print(f"[2] 已删除 {len(chunks)} 个块，{len(docs)} 篇文档重置为 PENDING")

    bump_corpus_version()

    # ---------- 3) 逐篇重建 ----------
    print(f"[3] 开始重建 {len(targets)} 篇文档 ...")
    import os

    ok = fail = skip = 0
    t_all = time.perf_counter()
    total_chars = 0

    for i, t in enumerate(targets, start=1):
        if not os.path.exists(t["path"]):
            skip += 1
            continue
        async with AsyncSessionLocal() as session:
            st = await ingest_document(session, t["id"])
        if st.get("ok"):
            ok += 1
            total_chars += st.get("chars", 0)
        else:
            fail += 1
            print(f"    [FAIL] doc_id={t['id']} {t['title'][:30]}: {st.get('error')}")
        if i % 50 == 0:
            cost = time.perf_counter() - t_all
            print(f"    进度 {i}/{len(targets)}  成功={ok} 失败={fail} "
                  f"耗时={cost:.0f}s  速度={i/max(cost,1e-6):.2f} docs/s", flush=True)

    cost = time.perf_counter() - t_all

    # ---------- 4) 核对 ----------
    print()
    print("=" * 80)
    print("重建完成")
    print("=" * 80)
    print(f"  成功 {ok} / 失败 {fail} / 跳过(源文件缺失) {skip}")
    print(f"  总耗时 {cost:.1f}s（{ok / max(cost, 1e-6):.2f} 篇/秒）")
    print(f"  向量总数: {await store.count()}")

    async with AsyncSessionLocal() as session:
        chunks = (await session.execute(
            select(DocumentChunk).where(DocumentChunk.is_active == True)  # noqa: E712
        )).scalars().all()
        models = {}
        for c in chunks:
            models[c.embedding_model] = models.get(c.embedding_model, 0) + 1
        print(f"  块总数: {len(chunks)}（子块 {sum(1 for c in chunks if not c.is_parent)}）")
        print(f"  块上记录的 embedding_model 分布: {models}")
        bad = [c for c in chunks if c.embedding_model != emb.name]
        if bad:
            print(f"  [警告] {len(bad)} 个块的模型名与新模型不一致")
        else:
            print("  所有块的 embedding_model 已统一为新模型 ✓")


asyncio.run(main())
