"""
把合成语料的部门归属隔离出去（让演示数据保持干净）。

问题（演示时发现的，经历了两次修正，是很好的"数据管理"教训）：
    第 1 轮：生成合成语料时 department 写成 ""（全公司可见），
        于是张三（研发/密级2）问"一线城市住宿标准"，命中了十几个部门的报销标准，
        大模型答"各部门规定不一致"—— 语义上没错，但演示效果很差。
    第 2 轮：按文件名前缀把合成语料分到各部门（rd_ -> rd）。
        结果张三**一样**被污染：他自己部门（rd）就有 30 篇合成文档，
        其中也有"差旅与住宿报销标准"，于是正确答案和合成数据混在一起。
    根因：合成语料和演示语料**共用同一个可见空间**，而且主题重叠
        （给合成语料用的模板里就有"差旅与住宿报销标准"）。

最终方案：让合成语料对演示账号完全不可见。
    合成语料的部门统一设为 SYNTHETIC_DEPT = "__corpus__"，
    它不属于任何真实用户部门，于是 perm.can_access() 会把它们过滤掉：
      - 演示账号（rd/finance/it）只看到原始 3 篇干净文档 -> 演示效果好；
      - 管理员（is_admin=True）不受部门限制 -> 评测仍能在 300+ 篇大语料上跑。
    这个"演示环境与评测环境在数据上隔离"的做法本身也值得讲：
    评测需要规模和噪声，演示需要干净可解释，两者目标冲突，不该混在一起。

用法：
    .venv\\Scripts\\python.exe scripts/fix_corpus_departments.py --dry-run
    .venv\\Scripts\\python.exe scripts/fix_corpus_departments.py
    .venv\\Scripts\\python.exe scripts/fix_corpus_departments.py --restore   # 恢复成按部门归属
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import select  # noqa: E402

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.models import Document, DocumentChunk  # noqa: E402
from app.vectorstore.base import get_vector_store  # noqa: E402

logger = get_logger("fix-dept")

# 合成语料专用部门标识：不属于任何真实部门，只有管理员能看见
SYNTHETIC_DEPT = "__corpus__"

PREFIX_TO_DEPT = {
    "rd": "rd", "mkt": "mkt", "finance": "finance", "hr": "hr", "scm": "scm",
    "cs": "cs", "legal": "legal", "data": "data", "product": "product", "qa": "qa",
}

SEED_TITLES = {
    "员工休假与考勤管理制度",
    "研发中心技术规范与故障处理手册",
    "2024年度财务预算与费用审批权限规定",
}


def infer_dept(filename: str, restore: bool) -> str | None:
    prefix = filename.split("_")[0].lower()
    if prefix not in PREFIX_TO_DEPT:
        return None
    return PREFIX_TO_DEPT[prefix] if restore else SYNTHETIC_DEPT


async def main(dry_run: bool = False, restore: bool = False) -> None:
    setup_logging("WARNING")
    await init_db()
    store = await get_vector_store()

    async with AsyncSessionLocal() as session:
        docs = (await session.execute(
            select(Document).where(Document.is_active == True)  # noqa: E712
        )).scalars().all()

        targets = []
        for d in docs:
            if d.title in SEED_TITLES:
                continue
            dept = infer_dept(d.filename, restore)
            if dept and (d.department or "") != dept:
                targets.append((d, dept))

        mode = "恢复按部门归属" if restore else f"隔离为 {SYNTHETIC_DEPT}"
        print("=" * 84)
        print(f"{mode} —— 待修改 {len(targets)} 篇 / 全部 {len(docs)} 篇"
              + ("（dry-run）" if dry_run else ""))
        print("=" * 84)
        if dry_run:
            for d, dept in targets[:8]:
                print(f"  doc_id={d.id:<4} {d.filename[:44]:<46} -> {dept}")
            print(f"  ... 共 {len(targets)} 篇")
            return

        ok = 0
        for i, (d, dept) in enumerate(targets, start=1):
            d.department = dept
            chunks = (await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == d.id)
            )).scalars().all()
            for c in chunks:
                c.department = dept

            # 向量库 metadata 是权限过滤的实际依据，必须同步更新
            #（否则会出现"库里允许、向量库不许"的不一致）
            child_uids = [c.chunk_uid for c in chunks if not c.is_parent]
            if child_uids:
                try:
                    await asyncio.to_thread(
                        store._collection.update,
                        ids=child_uids,
                        metadatas=[{"department": dept} for _ in child_uids],
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"更新向量库 metadata 失败 doc_id={d.id}: {e}")

            ok += 1
            if i % 50 == 0:
                await session.commit()
                print(f"  已处理 {i}/{len(targets)}", flush=True)

        await session.commit()
        print(f"\n完成：{ok} 篇已更新（文档表 + 块表 + 向量库 metadata）")
        if not restore:
            print(f"  合成语料现在只对管理员可见（department={SYNTHETIC_DEPT}），")
            print("  演示账号（rd/finance/it）将恢复只看到原始 3 篇干净文档。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="隔离/恢复合成语料的部门归属")
    p.add_argument("--dry-run", action="store_true", help="只统计不修改")
    p.add_argument("--restore", action="store_true", help="恢复成按文件名前缀的部门归属")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(main(dry_run=args.dry_run, restore=args.restore))
