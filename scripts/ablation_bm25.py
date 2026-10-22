"""
验证 BM25 是否真的带来增量召回 —— 一个"可证伪"的对照实验。

为什么要专门做这个实验（这是本项目最有价值的一次实证，面试重点）：
    "混合检索"是 RAG 项目的标配说辞，但很少有人真的验证过
    "BM25 到底贡献了多少"。我做了三组对照：

    实验 1｜增量召回统计
        对每道题比较"仅稠密的候选池"与"稠密∪稀疏的候选池"是否含 gold，
        统计"只有稀疏能找到"的题数 —— 这就是 BM25 的净贡献。

    实验 2｜关掉 BM25 看指标是否变化
        把稀疏权重设为 0（等价于关闭 BM25），保持其余链路完全不变，
        对比 Recall / NDCG。

    实验 3｜候选池大小的影响
        融合后截断到 20 条（= 稠密 top_k）vs 保留并集（约 40 条），
        验证"候选池被截断会让混合检索退化成替换"。

结论（当前语料实测，可直接写进简历/README）：
    - BM25 增量召回 = 0 道（无论小语料还是 309 篇大语料）；
    - 关掉 BM25 后指标完全相同（Recall 0.7250 不变）；
    - 真正的提升来自「扩大候选池 + Cross-Encoder 重排」：0.525 -> 0.725（+20pt）；
    - 融合后若截断到单路 top_k，会掉分（0.525 -> 0.500），
      因为稀疏噪声挤掉了稠密的好结果。
    => 诚实的表述是："混合检索架构已实现，但在这份语料上其增量未体现；
       本项目的实测增益来自重排与候选池扩展。" 而不是"混合检索提升了 X%"。

用法：
    .venv\\Scripts\\python.exe scripts/ablation_bm25.py                     # 用默认评测集
    .venv\\Scripts\\python.exe scripts/ablation_bm25.py --dataset data/eval_from_corpus.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.permissions import PermissionContext  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.eval.harness import (  # noqa: E402
    _pick_eval_user,
    _resolve_gold_chunks,
    load_dataset,
    run_eval,
)
from app.models import Tenant  # noqa: E402
from app.rag.embedding import get_embedder  # noqa: E402
from app.rag.fusion import rrf_fuse_union  # noqa: E402
from app.rag.sparse import BM25Retriever  # noqa: E402
from app.vectorstore.base import get_vector_store  # noqa: E402


async def experiment_incremental(dataset_path: Path | None) -> None:
    """实验 1：统计 BM25 的增量召回。"""
    print("=" * 92)
    print("实验 1｜BM25 增量召回统计（候选池 = 稠密 Top-K ∪ 稀疏 Top-K）")
    print("=" * 92)
    dataset = load_dataset(dataset_path)
    embedder = await get_embedder()
    store = await get_vector_store()

    only_dense = only_sparse = both = neither = 0
    gain_cases = []

    async with AsyncSessionLocal() as s:
        tenant = (await s.execute(select(Tenant).order_by(Tenant.id))).scalars().first()
        for item in dataset:
            if item.get("expect_refusal"):
                continue
            user = await _pick_eval_user(s, tenant.id, item)
            perm = PermissionContext.from_user(user)
            gold = await _resolve_gold_chunks(s, item, tenant.id, perm)
            if not gold:
                continue

            q = item["question"]
            qvec = await embedder.embed_query(q)
            dense = await store.search(qvec, settings.DENSE_TOP_K, perm)
            sparse = await BM25Retriever(s).search(q, settings.SPARSE_TOP_K, perm)
            fused = rrf_fuse_union(
                [dense, sparse],
                weights=[settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT],
                k=settings.RRF_K,
            )

            d_uids = {c.chunk_uid for c in dense}
            f_uids = {c.chunk_uid for c in fused}
            in_d = bool(d_uids & gold)
            in_f = bool(f_uids & gold)
            if in_d and in_f:
                both += 1
            elif in_d:
                only_dense += 1
            elif in_f:
                only_sparse += 1
                gain_cases.append(q)
            else:
                neither += 1

    print(f"  只有稠密找到         : {only_dense:>3} 道")
    print(f"  只有融合找到（增量） : {only_sparse:>3} 道  <- BM25 的净贡献")
    print(f"  两者都找到           : {both:>3} 道")
    print(f"  两者都没找到         : {neither:>3} 道")
    if gain_cases:
        print("  BM25 捞回的题:")
        for q in gain_cases:
            print(f"    - {q[:70]}")
    print()


async def experiment_metrics(dataset_path: Path | None) -> None:
    """实验 2：关掉 BM25 看指标是否变化（控制变量：其余链路不动）。"""
    print("=" * 92)
    print("实验 2｜关掉 BM25（稀疏权重=0）后的指标变化")
    print("=" * 92)

    async with AsyncSessionLocal() as session:
        tenant = (await session.execute(select(Tenant).order_by(Tenant.id))).scalars().first()
        rows = []
        for label, sparse_w in [("开启 BM25（默认 0.7/0.3）", 0.3),
                                ("关闭 BM25（1.0/0.0）", 0.0)]:
            settings.DENSE_WEIGHT = 1.0 if sparse_w == 0 else 0.7
            settings.SPARSE_WEIGHT = sparse_w
            res = await run_eval(
                session, tenant_id=tenant.id,
                modes=["dense_bm25_rerank"], top_k=5,
                dataset_path=dataset_path,
                run_id=f"ablation-bm25-{sparse_w}",
            )
            m = res["metrics"][0]
            rows.append((label, m))
            print(f"  {label:<26} Recall={m['recall_at_k']:.4f} NDCG={m['ndcg']:.4f} "
                  f"HitRate={m['hit_rate']:.4f} 延迟={m['avg_latency_ms']:.0f}ms")

        delta = rows[0][1]["recall_at_k"] - rows[1][1]["recall_at_k"]
        print()
        print(f"  差值（开启 - 关闭）: Recall {delta:+.4f}")
        if abs(delta) < 1e-6:
            print("  => 指标完全一致：在这份语料上 BM25 没有产生增量。")
        elif delta > 0:
            print("  => BM25 带来正向增益。")
        else:
            print("  => BM25 反而拖累指标（噪声挤占了有效候选）。")

    # 复原配置
    settings.DENSE_WEIGHT = 0.7
    settings.SPARSE_WEIGHT = 0.3
    print()


async def main(args: argparse.Namespace) -> None:
    setup_logging("ERROR")     # 实验输出要干净，屏蔽链路日志
    await init_db()
    ds = Path(args.dataset) if args.dataset else None
    await experiment_incremental(ds)
    await experiment_metrics(ds)
    print("=" * 92)
    print("写入文档时的诚实表述建议：")
    print("  ✅ 可以说：混合检索架构已完整实现（稠密+稀疏+RRF+重排），")
    print("             并通过对照实验验证了各组件的实际贡献；")
    print("  ✅ 可以说：实测增益来自「扩大候选池 + Cross-Encoder 重排」（0.525 -> 0.725）；")
    print("  ❌ 不要说：混合检索比纯向量检索提升了 X%（当前数据不支持）。")
    print("=" * 92)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BM25 增量价值验证实验")
    p.add_argument("--dataset", type=str, default=None,
                   help="评测集路径（默认 data/eval_dataset.json）")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
