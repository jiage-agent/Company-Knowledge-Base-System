"""标定本地 Cross-Encoder 重排的阈值 + 实测延迟。

为什么要单独标定（面试可以讲）：
    阈值不是"能力的属性"，而是"某个模型在某个语料上的分数分布的分界线"。
    换重排模型（阿里云 gte-rerank-v2 -> 本地 bge-reranker-base）后，
    分数的尺度和分布完全不同：
      - 阿里云 API 返回归一化相关性，无关内容 < 0.02；
      - 本地 bge-reranker-base 输出 logits，经 sigmoid 后
        连不相关内容也可能在 0.9 以上（模型对"格式相似"的文本都给高分）。
    如果沿用旧阈值 0.115，等于完全不过滤 —— 必须重新标定。
"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, r"D:\develop\Company-Knowledge-Base-System")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sqlmodel import select

from app.core.permissions import PermissionContext
from app.db.session import AsyncSessionLocal, init_db
from app.eval.harness import _pick_eval_user, _resolve_gold_chunks, load_dataset
from app.models import Tenant
from app.rag.embedding import get_embedder
from app.rag.reranker import LocalCrossEncoderReranker
from app.vectorstore.base import get_vector_store


async def main():
    await init_db()
    dataset = load_dataset()
    emb = await get_embedder()
    store = await get_vector_store()

    rr = LocalCrossEncoderReranker()
    t0 = time.perf_counter()
    model = rr._ensure_model()
    print(f"模型加载: {time.perf_counter()-t0:.1f}s  device={rr._device}  "
          f"可用={model is not None}")

    answerable, refusal = [], []
    latencies = []

    async with AsyncSessionLocal() as s:
        tenant = (await s.execute(select(Tenant).order_by(Tenant.id))).scalars().first()

        for item in dataset:
            user = await _pick_eval_user(s, tenant.id, item)
            perm = PermissionContext.from_user(user)
            gold = await _resolve_gold_chunks(s, item, tenant.id, perm)
            if not item.get("expect_refusal") and not gold:
                continue

            q = item["question"]
            qvec = await emb.embed_query(q)
            cands = await store.search(qvec, 30, perm)   # 模拟 RERANK_TOP_N=30
            if not cands:
                continue

            t1 = time.perf_counter()
            scored = await rr.rerank(q, cands, top_k=len(cands))
            latencies.append((time.perf_counter() - t1) * 1000)

            # 看"相关块"与"不相关块"的分数差异
            if item.get("expect_refusal"):
                refusal.append((max(sc.score for sc in scored), q[:40]))
            else:
                rel = [sc.score for sc in scored if sc.chunk_uid in gold]
                irr = [sc.score for sc in scored if sc.chunk_uid not in gold]
                if rel:
                    answerable.append((max(rel), q[:40]))
                if irr:
                    refusal.append((max(irr), "[不相关块] " + q[:36]))

    def stat(name, xs):
        if not xs:
            print(f"  {name}: 无样本")
            return
        v = [x[0] for x in xs]
        print(f"  {name:<12} n={len(v):<3} min={min(v):.4f} p25={sorted(v)[len(v)//4]:.4f} "
              f"中位={statistics.median(v):.4f} p75={sorted(v)[3*len(v)//4]:.4f} max={max(v):.4f}")

    print()
    print("=" * 78)
    print("本地重排分数分布（bge-reranker-base, sigmoid 后）")
    print("=" * 78)
    stat("相关块", answerable)
    stat("不相关块", refusal)

    if answerable and refusal:
        a = sorted(x[0] for x in answerable)
        r = sorted(x[0] for x in refusal)
        print()
        print(f"  相关块最低分 = {a[0]:.4f}")
        print(f"  不相关块最高分 = {r[-1]:.4f}")
        print(f"  重叠情况: {'完全重叠（单阈值不可分）' if r[-1] >= a[0] else '可分'}")
        # 找最佳阈值
        best = (0, -1)
        for i in range(50, 100):
            th = i / 100
            correct = sum(1 for x in a if x >= th) + sum(1 for x in r if x < th)
            if correct > best[1]:
                best = (th, correct)
        total = len(a) + len(r)
        print(f"  最佳阈值 ≈ {best[0]:.2f}  分开准确率 {best[1]}/{total} = {best[1]/total:.1%}")

    print()
    print("=" * 78)
    print("延迟（CPU）")
    print("=" * 78)
    if latencies:
        print(f"  每次重排 30 条候选: 中位 {statistics.median(latencies):.0f}ms  "
              f"max {max(latencies):.0f}ms  共 {len(latencies)} 次")


asyncio.run(main())
