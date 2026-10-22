"""
合成语料生成器：把演示语料扩到"BM25 能体现实力"的规模。

为什么必须扩语料（这是实测得出的结论，也是本项目最有价值的一次实验）：
    在 16 个向量的小语料上做了两组对照实验：
      ① 关掉 BM25（稀疏权重=0）vs 打开 BM25 -> 指标**完全相同**（Recall 1.0 / NDCG 0.9902）
      ② 统计 BM25 的增量召回 -> **0 道**（68 道题里稠密的候选池已全部包含答案）
    根因：语料只有 16 个向量，而 DENSE_TOP_K=20 ——
    稠密检索一次就把全部内容取回来了，BM25 没有发挥空间，
    它只是在改排序，而排序恰好被重排接管了。
    **小语料上永远测不出混合检索的价值，这是评测设计的规模问题，不是实现问题。**

    真实企业知识库是成千上万篇文档。语料一大，两件事同时发生：
      1. 稠密检索的 Top-20 不再包含全部内容 -> 稀疏检索才有机会贡献增量；
      2. 编号/金额/专有名词这类"字面特征"在纯语义向量里容易被稀释 ->
         BM25 的精确匹配优势才显现。
    所以这里合成一批结构相似但内容不同的文档，把语料拉到能体现差异的规模。

生成策略（为什么让模型"按模板造"而不是自由发挥）：
    评测需要的是"同一类条款散落在大量文档里"的干扰环境，
    所以给定章节模板（年假/报销/审批/工单……），
    让模型只替换具体数值与措辞，保证：
      - 有大量"语义相似但答案不同"的干扰文档（正是稠密检索容易搞错的场景）
      - 有明确的编号与数字（正是 BM25 擅长的场景）
      - 内容可控、可复现，且不涉及任何真实企业数据

用法：
    .venv\\Scripts\\python.exe scripts/generate_corpus.py --count 150
    .venv\\Scripts\\python.exe scripts/generate_corpus.py --count 150 --ingest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import select  # noqa: E402

from app.config import BASE_DIR  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.models import Document, Tenant, User  # noqa: E402
from app.rag.ingest import (  # noqa: E402
    create_document,
    ingest_document,
    save_upload,
    sha256_of,
)
from app.rag.llm import chat  # noqa: E402

logger = get_logger("corpus")

# 并发度：太小太慢，太大容易触发接口限流（实测 4 路稳定）
CONCURRENCY = 4

# 部门 / 城市 / 币种等维度，用于制造"条款结构相同、数值不同"的干扰文档
DEPARTMENTS = [
    ("研发中心", "rd"), ("市场部", "mkt"), ("财务部", "finance"), ("人力资源部", "hr"),
    ("供应链中心", "scm"), ("客户成功部", "cs"), ("法务部", "legal"), ("数据平台部", "data"),
    ("产品部", "product"), ("质量保障部", "qa"),
]
TOPICS = [
    ("差旅与住宿报销标准", "policy", "出差住宿标准、交通标准、报销时限"),
    ("加班与调休管理办法", "policy", "加班申请流程、调休折算比例、调休有效期"),
    ("采购与付款审批流程", "policy", "采购金额分级审批、付款账期、预付款限制"),
    ("数据安全与权限管理规范", "manual", "数据分级、访问审批、外发管控"),
    ("设备与资产领用制度", "policy", "设备申领流程、折旧年限、离职归还"),
    ("绩效考核与晋升办法", "policy", "考核周期、评级比例、晋升条件"),
    ("供应商准入与评估规范", "manual", "准入材料、评估指标、年审要求"),
    ("客户投诉处理流程", "manual", "响应时限、升级路径、赔付标准"),
    ("信息安全事件响应预案", "manual", "事件分级、上报时限、处置流程"),
    ("培训与认证管理办法", "policy", "培训审批、费用承担、认证奖励"),
    ("岗位职级与薪酬结构说明", "report", "职级体系、薪酬带宽、调薪周期"),
    ("固定资产盘点规程", "manual", "盘点周期、差异处理、责任认定"),
    ("差旅补贴与市内交通细则", "policy", "补贴标准、票据要求、超标处理"),
    ("合同评审与用印管理规范", "contract", "评审节点、用印审批、归档要求"),
    ("研发项目立项与验收流程", "manual", "立项条件、里程碑、验收标准"),
]

PROMPT_TEMPLATE = """你是企业制度文档撰写助手。请为「{dept}」编写一份《{topic}》，文档类型为{biz}。

要求：
1. 分成 4~5 个章节，每章 2~4 条；条款要具体，包含**明确的数字**（金额、天数、比例、时限、编号格式）。
2. 全文 600~900 字，使用中文，条款编号统一用"第X条"。
3. 数字要符合这个部门的业务特点（{dept}），不要出现"根据实际情况""另行规定"这类空话。
4. 其中至少包含一个**编号格式**规定（例如申请单编号、工单编号、合同编号的格式），例如 XXX-年份-四位流水号。
5. 直接输出文档正文，不要任何解释、不要 markdown 代码块。

参考覆盖内容：{hint}
"""


async def generate_one(dept: str, topic: str, biz: str, hint: str, retry: int = 2) -> str:
    prompt = PROMPT_TEMPLATE.format(dept=dept, topic=topic, biz=biz, hint=hint)
    for attempt in range(retry + 1):
        try:
            text = await chat([{"role": "user", "content": prompt}],
                              temperature=0.9, max_tokens=2000)
            text = text.strip()
            if len(text) > 300:
                return text
        except Exception as e:  # noqa: BLE001
            logger.warning(f"生成失败（第 {attempt + 1} 次）: {str(e)[:120]}")
        await asyncio.sleep(1.0)
    return ""


async def generate(count: int, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # 幂等：已生成过的文件不重复生成（省 token）
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        for item in json.loads(manifest_path.read_text(encoding="utf-8")):
            existing[item["filename"]] = item

    combos: list[tuple[str, str, str, str, str]] = []
    for dept, dept_code in DEPARTMENTS:
        for topic, biz, hint in TOPICS:
            combos.append((dept, dept_code, topic, biz, hint))
    random.Random(42).shuffle(combos)          # 固定种子，保证可复现

    todo = combos[:count]
    results: list[dict] = list(existing.values())
    done_names = set(existing.keys())

    # 先把待生成清单算出来（跳过已存在的），再并发生成
    pending: list[tuple[str, str, str, str, str, str]] = []
    for dept, dept_code, topic, biz, hint in todo:
        filename = f"{dept_code}_{abs(hash((dept, topic))) % 100000:05d}_{topic}.txt"
        if filename in done_names:
            continue
        pending.append((dept, dept_code, topic, biz, hint, filename))

    print(f"  待生成 {len(pending)} 篇（已有 {len(done_names)} 篇），并发度 {CONCURRENCY}")

    # 并发生成：串行每篇约 6 秒，150 篇要 15 分钟；并发 4 路压到 4 分钟左右。
    # 用信号量而不是无脑 gather，避免触发接口限流（429）。
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done_count = 0

    async def worker(job: tuple[str, str, str, str, str, str]) -> None:
        nonlocal done_count
        dept, dept_code, topic, biz, hint, filename = job
        async with sem:
            text = await generate_one(dept, topic, biz, hint)
        async with lock:
            done_count += 1
            if not text:
                print(f"  [{done_count}/{len(pending)}] 《{dept} {topic}》 失败，跳过", flush=True)
                return
            (out_dir / filename).write_text(text, encoding="utf-8")
            results.append({
                "filename": filename, "dept": dept, "dept_code": dept_code,
                "topic": topic, "doc_type": biz, "chars": len(text),
            })
            print(f"  [{done_count}/{len(pending)}] 《{dept} {topic}》 OK ({len(text)} 字)", flush=True)

    await asyncio.gather(*(worker(j) for j in pending))

    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


async def ingest_all(items: list[dict], out_dir: Path) -> None:
    """把合成语料入库（去重靠 file_hash，重复执行不会重复入库）。"""
    await init_db()
    async with AsyncSessionLocal() as session:
        tenant = (await session.execute(select(Tenant).order_by(Tenant.id))).scalars().first()
        if tenant is None:
            print("没有租户，请先运行 scripts/seed.py")
            return
        admin = (await session.execute(
            select(User).where(User.tenant_id == tenant.id, User.role == "admin")
        )).scalars().first()
        if admin is None:
            print("没有管理员账号，请先运行 scripts/seed.py")
            return

        ok = skip = fail = 0
        for i, item in enumerate(items, start=1):
            src = out_dir / item["filename"]
            if not src.exists():
                continue
            # 部门映射到 ACL：合成文档按部门隔离，密级统一为 2（内部）
            dept_code = item["dept_code"]
            saved = save_upload(src.read_bytes(), item["filename"])
            doc = await create_document(
                session,
                tenant_id=tenant.id,
                user_id=admin.id,
                title=f"{item['dept']}{item['topic']}",
                filename=item["filename"],
                file_path=saved,
                file_hash=sha256_of(saved),
                doc_type=item["doc_type"],
                department="",              # 全公司可见，作为干扰语料参与检索
                security_level=1,
                allowed_roles=[],
            )
            if doc.status == "indexed":
                skip += 1
                continue
            stats = await ingest_document(session, doc.id)
            if stats.get("ok"):
                ok += 1
            else:
                fail += 1
                logger.warning(f"入库失败 {item['filename']}: {stats.get('error')}")
            if i % 20 == 0:
                print(f"  已处理 {i}/{len(items)}（成功 {ok} 跳过 {skip} 失败 {fail}）", flush=True)

        print(f"\n入库完成: 成功 {ok} / 跳过(已存在) {skip} / 失败 {fail}")
        total = len((await session.execute(
            select(Document).where(Document.is_active == True)  # noqa: E712
        )).scalars().all())
        print(f"当前文档总数: {total}")


async def main(args: argparse.Namespace) -> None:
    setup_logging("WARNING")
    out_dir = BASE_DIR / "data" / "corpus"
    print("=" * 88)
    print(f"合成语料生成：目标 {args.count} 篇，输出到 {out_dir}")
    print("=" * 88)
    items = await generate(args.count, out_dir)
    print(f"\n共 {len(items)} 篇语料就绪")
    if args.ingest:
        print("\n开始入库（会调用 embedding 接口，按量计费）...")
        await ingest_all(items, out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成合成企业文档语料")
    p.add_argument("--count", type=int, default=150, help="生成篇数（模板组合上限 150）")
    p.add_argument("--ingest", action="store_true", help="生成后直接入库")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
