"""
一键灌入演示数据。

用法：
    .venv\\Scripts\\python.exe scripts/seed.py          # 幂等：已存在则跳过
    .venv\\Scripts\\python.exe scripts/seed.py --reset  # 清空向量库后重建

为什么要有这个脚本（面试可讲）：
    1. 演示可靠性：面试前一条命令就能把环境恢复到"有数据、可提问"的状态，
       不依赖手工上传，避免现场翻车；
    2. 可复现性：种子数据固定，评测结果才能在多次运行间对比；
    3. 权限演示的基础：脚本创建了三种不同权限的账号 + 三种密级的文档，
       "同一问题不同账号答案不同"这个最有说服力的演示就靠它。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 让脚本能直接运行（把项目根目录加入 sys.path）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlmodel import select  # noqa: E402

from app.config import BASE_DIR  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.db.session import AsyncSessionLocal, init_db  # noqa: E402
from app.models import Document, Tenant, User  # noqa: E402
from app.rag.ingest import (  # noqa: E402
    create_document,
    ingest_document,
    save_upload,
    sha256_of,
)
from app.seed_data import SEED_DOCS, SEED_USERS  # noqa: E402
from app.vectorstore.base import get_vector_store  # noqa: E402

logger = get_logger("seed")


async def get_or_create_tenant(session, name: str) -> Tenant:
    tenant = (await session.execute(select(Tenant).where(Tenant.name == name))).scalars().first()
    if tenant:
        return tenant
    tenant = Tenant(name=name)
    session.add(tenant)
    await session.commit()
    await session.refresh(tenant)
    logger.info(f"创建租户: {name} (id={tenant.id})")
    return tenant


async def get_or_create_user(session, tenant_id: int, spec: dict) -> User:
    stmt = select(User).where(User.tenant_id == tenant_id, User.username == spec["username"])
    user = (await session.execute(stmt)).scalars().first()
    if user:
        return user
    user = User(
        tenant_id=tenant_id,
        username=spec["username"],
        hashed_password=hash_password(spec["password"]),
        display_name=spec["display_name"],
        role=spec["role"],
        department=spec["department"],
        security_level=spec["security_level"],
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info(f"创建用户: {spec['username']} / {spec['password']} "
                f"(角色={spec['role']} 部门={spec['department']} 密级={spec['security_level']})")
    return user


async def seed(reset: bool = False) -> None:
    setup_logging("INFO")
    await init_db()

    if reset:
        store = await get_vector_store()
        await store.reset()
        logger.warning("已清空向量库（--reset）")

    async with AsyncSessionLocal() as session:
        # ---------- 1. 租户与用户 ----------
        print("\n" + "=" * 72)
        print("【1/3】创建租户与用户")
        print("=" * 72)
        tenant_map: dict[str, Tenant] = {}
        for name in {d["tenant"] for d in SEED_DOCS} | {u["tenant"] for u in SEED_USERS}:
            tenant_map[name] = await get_or_create_tenant(session, name)

        user_map: dict[str, User] = {}
        for spec in SEED_USERS:
            tenant = tenant_map[spec["tenant"]]
            user_map[spec["username"]] = await get_or_create_user(session, tenant.id, spec)

        # ---------- 2. 文档入库 ----------
        print("\n" + "=" * 72)
        print("【2/3】解析并入库演示文档（会调用 embedding 接口）")
        print("=" * 72)
        admin = user_map["admin"]
        for spec in SEED_DOCS:
            tenant = tenant_map[spec["tenant"]]
            # 先检查同标题文档是否已入库（幂等）
            stmt = select(Document).where(
                Document.tenant_id == tenant.id,
                Document.title == spec["title"],
                Document.is_active == True,  # noqa: E712
            )
            exists = (await session.execute(stmt)).scalars().first()
            if exists and exists.status == "indexed":
                print(f"  [跳过] 《{spec['title']}》已入库 "
                      f"(doc_id={exists.id}, {exists.chunk_count} 块)")
                continue

            src = BASE_DIR / "data" / "_seed_src" / spec["filename"]
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(spec["content"], encoding="utf-8")

            saved = save_upload(spec["content"].encode("utf-8"), spec["filename"])
            doc = await create_document(
                session,
                tenant_id=tenant.id,
                user_id=admin.id,
                title=spec["title"],
                filename=spec["filename"],
                file_path=saved,
                file_hash=sha256_of(saved),
                doc_type=spec["doc_type"],
                department=spec["department"],
                security_level=spec["security_level"],
                allowed_roles=[],
            )
            t0 = time.perf_counter()
            stats = await ingest_document(session, doc.id)
            await session.refresh(doc)
            flag = "OK  " if stats.get("ok") else "FAIL"
            print(f"  [{flag}] 《{spec['title']}》 "
                  f"密级={spec['security_level']} 部门={spec['department'] or '全公司'} "
                  f"-> {doc.chunk_count} 个子块 / {stats.get('vectors', 0)} 向量 "
                  f"({time.perf_counter() - t0:.1f}s)")
            if not stats.get("ok"):
                print(f"         错误: {stats.get('error')}")

        # ---------- 3. 汇总 ----------
        print("\n" + "=" * 72)
        print("【3/3】种子数据完成")
        print("=" * 72)
        docs = (await session.execute(
            select(Document).where(Document.is_active == True)  # noqa: E712
        )).scalars().all()
        store = await get_vector_store()
        print(f"  文档数: {len(docs)}   向量数: {await store.count()}")
        print("\n  可用账号（租户 = acme）:")
        for spec in SEED_USERS:
            print(f"    {spec['username']:<10} / {spec['password']:<10} "
                  f"{spec['display_name']:<22} 角色={spec['role']:<7} "
                  f"部门={spec['department']:<8} 密级={spec['security_level']}")
        print("\n  权限演示提示：")
        print("    - zhangsan（研发/密级2）问「50万元以上的费用需要谁审批？」-> 应拒答（财务文档密级3）")
        print("    - lisi（财务/密级3）   问同一个问题 -> 应正常回答并给出财务文档引用")


def main() -> None:
    parser = argparse.ArgumentParser(description="灌入演示数据")
    parser.add_argument("--reset", action="store_true", help="先清空向量库再灌数据")
    args = parser.parse_args()
    asyncio.run(seed(reset=args.reset))


if __name__ == "__main__":
    main()
