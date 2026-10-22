"""
重建索引脚本。

什么时候需要跑它（这是真实运维场景，面试可以讲）：
    1. 改了切分策略（chunk_size / overlap / recursive->semantic）；
    2. 换了 embedding 模型（向量维度或语义空间变了，旧向量全部作废）；
    3. 升级了向量库（Chroma -> Milvus 迁移时需要把数据重新灌进去）；
    4. 索引损坏或不一致（比如向量库与关系库对不上）。

为什么"重建"比"增量修补"更常用：
    改切分策略会让所有块边界变化，增量修补无从下手；
    而重建的成本主要是 embedding 调用（有 API 费用与时间），
    所以生产系统通常会：先小批量试跑新策略 -> 用评测集对比 -> 确认更优再全量重建。
    本脚本支持 --document-id 参数，就是为了"先试一篇再全量"。

用法：
    python scripts/reindex.py                    # 重建全部文档
    python scripts/reindex.py --document-id 3    # 只重建某一份（试跑用）
    python scripts/reindex.py --dry-run          # 只列出将要重建的文档
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import select  # noqa: E402

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.models import DocStatus, Document, DocumentChunk  # noqa: E402
from app.rag.ingest import ingest_document  # noqa: E402
from app.vectorstore.base import get_vector_store  # noqa: E402

logger = get_logger("reindex")


async def purge_document_index(session, document_id: int) -> dict:
    """
    清掉一份文档的旧索引（关系库物理删除 + 向量库删除）。

    ⚠️ 这里为什么是【物理删除】而不是软删除（一个需要讲清楚的设计取舍）：

      我们的 chunk_uid 是**确定性**生成的：d{doc_id}v{version}c{序号}
      这样设计有三个好处：
        1. 幂等：重复入库不会产生重复向量；
        2. 可复现：评测集标注的 gold chunk_uid 在重建索引后依然有效
           （如果是随机 uid，每次重建评测基线就没法对比了）；
        3. 可定位：能按前缀精确找到某份文档的所有向量。

      但确定性 uid 与"软删除保留历史"是冲突的：
        重建索引时新块会生成完全相同的 uid，
        而 SQLite 的 UNIQUE 约束不允许两行同 uid（哪怕旧行已 is_active=False）。
      两种解法：
        A) uid 里加版本号 -> 每次重建 uid 都变，好处是可回溯，代价是评测基线失效；
        B) 物理删除旧块    -> uid 稳定，代价是丢失旧块文本（但源文件还在，可随时重建）。
      我们选 B：因为**评测可复现**对本项目更重要，
      而"历史可回溯"这个需求由 git 里的源文件 + 审计日志（audit_logs）覆盖了。
      真实生产里如果合规要求保留历史快照，就应当选 A，并给评测集做 uid 映射表 ——
      关键不是选哪个，而是知道各自的代价。

    顺序：先删向量库、再删关系库。
      最坏情况是"关系库还有记录但检索不到"（安全方向），
      而不是"向量还在但元数据没了"（会变成查不到权限的幽灵数据）。
    """
    store = await get_vector_store()
    deleted_vectors = await store.delete_by_document(document_id)

    stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    chunks = (await session.execute(stmt)).scalars().all()
    for c in chunks:
        await session.delete(c)
    await session.commit()
    return {"chunks_deleted": len(chunks), "vectors_deleted": deleted_vectors}


async def reindex(document_id: int | None = None, dry_run: bool = False) -> None:
    setup_logging("INFO")
    await init_db()

    # 第一段：只读取待重建清单，读完就关会话。
    # 为什么要分两段（实测踩坑）：向量库的删除是同步阻塞调用，
    # 需要丢进线程池执行（asyncio.to_thread）；如果同一个 AsyncSession
    # 在多次「线程池操作 + commit」之后继续使用，SQLAlchemy 的异步
    # 上下文会出现 greenlet 状态错乱（MissingGreenlet）。
    # 解决办法就是"每份文档一个短会话"——这也正是真实 worker 的做法：
    # 任务之间不共享会话，避免状态串味。
    async with AsyncSessionLocal() as session:
        stmt = select(Document).where(Document.is_active == True)  # noqa: E712
        if document_id:
            stmt = stmt.where(Document.id == document_id)
        docs = (await session.execute(stmt)).scalars().all()
        targets = [
            {"id": d.id, "title": d.title, "file_path": d.file_path,
             "file_type": d.file_type, "chunk_count": d.chunk_count}
            for d in docs
        ]

    if not targets:
        print("没有找到需要重建的文档")
        return

    print("=" * 78)
    print(f"待重建文档 {len(targets)} 份" + ("（dry-run，不会实际执行）" if dry_run else ""))
    print("=" * 78)
    for t in targets:
        exists = Path(t["file_path"]).exists()
        print(f"  doc_id={t['id']:<4} 《{t['title'][:32]}》 类型={t['file_type']} "
              f"旧块数={t['chunk_count']} 源文件={'存在' if exists else '【缺失!】'}")
    if dry_run:
        return

    ok, failed = 0, 0
    total_t0 = time.perf_counter()
    for t in targets:
        if not Path(t["file_path"]).exists():
            print(f"  [跳过] doc_id={t['id']} 源文件已丢失: {t['file_path']}")
            failed += 1
            continue

        print(f"\n  >>> 重建 doc_id={t['id']} 《{t['title'][:30]}》")
        # 每份文档独立会话：失败不影响其他文档，也不会有会话状态残留
        async with AsyncSessionLocal() as session:
            purge = await purge_document_index(session, t["id"])
            print(f"      清理旧索引: 删除块={purge['chunks_deleted']} "
                  f"删除向量={purge['vectors_deleted']}")

            doc = await session.get(Document, t["id"])
            doc.status = DocStatus.PENDING.value
            doc.chunk_count = 0
            await session.commit()

            t0 = time.perf_counter()
            stats = await ingest_document(session, t["id"])
            if stats.get("ok"):
                ok += 1
                print(f"      [OK] 新块数={stats.get('children')} 父块={stats.get('parents')} "
                      f"向量={stats.get('vectors')} 耗时={time.perf_counter() - t0:.1f}s")
            else:
                failed += 1
                print(f"      [FAIL] {stats.get('error')}")

    print("\n" + "=" * 78)
    print(f"重建完成：成功 {ok} 份，失败 {failed} 份，"
          f"总耗时 {time.perf_counter() - total_t0:.1f}s")
    store = await get_vector_store()
    print(f"当前向量总数: {await store.count()}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="重建文档索引")
    parser.add_argument("--document-id", type=int, default=None, help="只重建指定文档")
    parser.add_argument("--dry-run", action="store_true", help="只列出将重建的文档")
    args = parser.parse_args()
    asyncio.run(reindex(document_id=args.document_id, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
