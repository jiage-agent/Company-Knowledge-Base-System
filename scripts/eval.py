"""
消融实验脚本（命令行版）。

用法：
    .venv\\Scripts\\python.exe scripts/eval.py                      # 全量跑三组配置
    .venv\\Scripts\\python.exe scripts/eval.py --limit 8            # 只跑前 8 条（快速验证）
    .venv\\Scripts\\python.exe scripts/eval.py --modes dense_only dense_bm25
    .venv\\Scripts\\python.exe scripts/eval.py --save report.json   # 结果存文件，便于写进 README

为什么评测要能独立于 API 跑（面试可讲）：
    评测是"批量、可重复、可能要跑很久"的任务，不应该挂在 HTTP 请求里
    （会超时、会阻塞其他用户）。做成独立脚本后可以：
      1. 挂到 CI：每次改切分策略/换模型自动跑，防止效果悄悄退化；
      2. 本地反复调参：先 --limit 8 快速试，确认方向对了再全量跑；
      3. 结果落盘对比：把不同配置的报告存起来，形成"实验记录"，
         这比只记得"好像好一点"专业得多。
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

from app.core.logging import setup_logging  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.eval.harness import format_report, run_eval  # noqa: E402
from app.models import Tenant  # noqa: E402


async def main(args: argparse.Namespace) -> None:
    setup_logging("WARNING")       # 评测时降低日志级别，避免刷屏
    await init_db()

    async with AsyncSessionLocal() as session:
        # 默认用第一个租户（演示环境是 acme）
        tenant = (await session.execute(select(Tenant).order_by(Tenant.id))).scalars().first()
        if tenant is None:
            print("没有租户数据，请先运行 scripts/seed.py")
            return

        result = await run_eval(
            session,
            tenant_id=tenant.id,
            modes=args.modes,
            top_k=args.top_k,
            limit=args.limit,
            run_id=args.run_id,
            dataset_path=Path(args.dataset) if args.dataset else None,
        )

    print(format_report(result))

    # 逐题明细：定位"哪些题没召回对"，这是调优的入口
    if args.verbose:
        print("\n逐题明细（按 NDCG 升序，最差的排前面）:")
        print("-" * 100)
        for d in sorted(result["detail"], key=lambda x: x["ndcg"]):
            flag = "拒答题" if d["expect_refusal"] else "检索题"
            print(f"  [{d['mode'][:18]:<18}] {flag} ndcg={d['ndcg']:.2f} "
                  f"recall={d['recall']:.2f} 命中={d['retrieved']}条 gold={d['gold_count']}条 "
                  f"{d['latency_ms']:>5}ms")
            print(f"      问: {d['question']}")
            print(f"      命中来源: {d['top_titles']}")

    if args.save:
        out = Path(args.save)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整结果已保存: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG 消融实验")
    p.add_argument("--modes", nargs="+",
                   default=["dense_only", "dense_bm25", "dense_bm25_rerank"],
                   choices=["dense_only", "dense_bm25", "dense_bm25_rerank"],
                   help="要对比的配置组合")
    p.add_argument("--top-k", type=int, default=5, help="评估的 Top-K")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    p.add_argument("--save", type=str, default=None, help="结果保存到 JSON 文件")
    p.add_argument("--verbose", action="store_true", help="输出逐题明细")
    p.add_argument("--run-id", type=str, default=None, help="自定义实验编号")
    p.add_argument("--dataset", type=str, default=None,
                   help="评测集路径（默认 data/eval_dataset.json；"
                        "可用 data/eval_from_corpus.json 跑自动生成的可验证题）")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
