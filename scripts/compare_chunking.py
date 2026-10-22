"""
分块策略对比实验：recursive vs semantic。

为什么单独做这个实验（面试会问"你怎么选切分策略的"）：
  切分是 RAG 效果最关键的环节，但"语义切分更好"不能靠感觉断言 ——
  语义切分每篇文档都要多花约 2 秒（要向量化每个句子），
  这个代价必须用指标来证明值不值。

实验设计（控制变量）：
  同一批文档、同一批评测题、同一套检索配置（仅切分策略不同）：
    1. 用 recursive 重建索引 -> 跑评测 -> 记录指标
    2. 用 semantic  重建索引 -> 跑评测 -> 记录指标
    3. 对比 Recall / NDCG / HitRate 与索引耗时
  这是标准的 A/B 对照，避免"换了策略又换了模型"导致归因错误。

用法：
    .venv\\Scripts\\python.exe scripts/compare_chunking.py
    .venv\\Scripts\\python.exe scripts/compare_chunking.py --limit 20   # 快速试
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.eval.harness import run_eval  # noqa: E402
from app.models import DocStatus, Document, DocumentChunk, Tenant  # noqa: E402
from app.rag.ingest import ingest_document  # noqa: E402
from app.vectorstore.base import get_vector_store  # noqa: E402

logger = get_logger("compare")


async def rebuild_all(strategy: str) -> dict:
    """
    用指定策略重建全部文档索引，返回索引耗时与块数统计。

    注意：这里会物理删除旧块并重算向量 —— 所以要显式记录耗时，
    而这个耗时正是"语义切分的代价"里最实的一部分。
    """
    settings.CHUNK_STRATEGY = strategy
    store = await get_vector_store()

    async with AsyncSessionLocal() as session:
        docs = (await session.execute(
            select(Document).where(Document.is_active == True)  # noqa: E712
        )).scalars().all()
        targets = [{"id": d.id, "title": d.title, "path": d.file_path} for d in docs]

    total_sec = 0.0
    child_total = parent_total = 0

    for t in targets:
        if not Path(t["path"]).exists():
            continue
        # 每份文档一个短会话：避免向量库线程操作 + commit 之后会话状态错乱
        async with AsyncSessionLocal() as session:
            await store.delete_by_document(t["id"])
            old = (await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == t["id"])
            )).scalars().all()
            for c in old:
                await session.delete(c)
            doc = await session.get(Document, t["id"])
            doc.status = DocStatus.PENDING.value
            doc.chunk_count = 0
            await session.commit()

            t0 = time.perf_counter()
            stats = await ingest_document(session, t["id"])
            cost = time.perf_counter() - t0
            total_sec += cost
            child_total += stats.get("children", 0)
            parent_total += stats.get("parents", 0)
            logger.info(f"[{strategy}] {t['title'][:22]} 子块={stats.get('children')} "
                        f"父块={stats.get('parents')} {cost:.1f}s")

    return {
        "strategy": strategy,
        "index_seconds": round(total_sec, 2),
        "children": child_total,
        "parents": parent_total,
        "vectors": await store.count(),
    }


async def main(args: argparse.Namespace) -> None:
    setup_logging("WARNING")
    await init_db()

    results = []
    for strategy in ["recursive", "semantic"]:
        print("\n" + "=" * 92)
        print(f"【重建索引】策略 = {strategy}")
        print("=" * 92)
        idx = await rebuild_all(strategy)
        print(f"  父块 {idx['parents']} 个 / 子块 {idx['children']} 个 / 向量 {idx['vectors']} 条 "
              f"/ 索引耗时 {idx['index_seconds']}s")

        print(f"\n【评测】策略 = {strategy}  共 {args.limit or '全部'} 条")
        async with AsyncSessionLocal() as session:
            tenant = (await session.execute(select(Tenant).order_by(Tenant.id))).scalars().first()
            ev = await run_eval(
                session, tenant_id=tenant.id,
                modes=args.modes, top_k=args.top_k, limit=args.limit,
                run_id=f"chunk-{strategy}",
            )
        results.append({"index": idx, "eval": ev})
        for m in ev["metrics"]:
            print(f"  {m['mode']:<20} Recall={m['recall_at_k']:.4f} NDCG={m['ndcg']:.4f} "
                  f"HitRate={m['hit_rate']:.4f} 延迟={m['avg_latency_ms']:.0f}ms")

    # ---------- 对比汇总 ----------
    print("\n" + "=" * 92)
    print("分块策略对比（同文档、同评测题、同检索配置，仅切分策略不同）")
    print("=" * 92)
    print(f"{'策略':<12} {'子块数':>7} {'平均块长':>9} {'索引耗时':>9} "
          f"{'Recall@K':>10} {'NDCG':>8} {'HitRate':>9}")
    print("-" * 92)
    for r in results:
        idx, ev = r["index"], r["eval"]
        best = max(ev["metrics"], key=lambda m: m["ndcg"])
        avg_len = 0
        if idx["children"]:
            avg_len = 0  # 需要另算，这里只列块数
        print(f"{idx['strategy']:<12} {idx['children']:>7} {'-':>9} {idx['index_seconds']:>8.1f}s "
              f"{best['recall_at_k']:>10.4f} {best['ndcg']:>8.4f} {best['hit_rate']:>9.4f}")
    print("-" * 92)
    print("说明：指标取该策略下表现最好的检索配置；索引耗时是切分策略带来的真实代价。")

    if args.save:
        out = Path(args.save)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整结果已保存: {out.resolve()}")

    # 复原成配置文件里的策略，避免影响后续使用
    settings.CHUNK_STRATEGY = "recursive"
    print("\n提示：实验结束后索引停留在 semantic 策略上，"
          "如需恢复请重跑 scripts/reindex.py（或把 .env 的 CHUNK_STRATEGY 改回 recursive）")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="分块策略对比实验")
    p.add_argument("--modes", nargs="+", default=["dense_bm25_rerank"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条评测题（0=全量）")
    p.add_argument("--save", type=str, default="data/compare_chunking.json")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
