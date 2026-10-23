"""演示前的一键自检脚本。

面试现场只需要跑这一条命令，它会把"能不能演示"全部验证一遍：
  1. 服务是否在跑（不在跑就自动拉起）
  2. /warm 预热（向量化 + 重排 + 检索链路）
  3. 三个演示账号能否登录
  4. 权限隔离是否生效（张三看不到财务文档 / 李四能看到）
  5. 三类典型问题是否答对（事实 / 无权 / 超纲拒答）
全部通过才会输出 "READY FOR DEMO"。

用法：
    .venv\\Scripts\\python.exe scripts\\preflight.py
"""
from __future__ import annotations

import subprocess
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
PROJECT = r"D:\develop\Company-Knowledge-Base-System"

PASS, FAIL, WARN = "[OK]  ", "[FAIL]", "[WARN]"
problems: list[str] = []


def hr(t: str) -> None:
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def check_service() -> bool:
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def start_service() -> bool:
    print("  服务未运行，尝试启动 ...")
    subprocess.Popen(
        [f"{PROJECT}\\.venv\\Scripts\\python.exe", "-m", "uvicorn",
         "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        cwd=PROJECT,
        stdout=open(f"{PROJECT}\\logs\\preflight.out.log", "a"),
        stderr=open(f"{PROJECT}\\logs\\preflight.err.log", "a"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for i in range(60):
        time.sleep(2)
        if check_service():
            print(f"  启动成功（约 {(i + 1) * 2}s）")
            return True
    return False


def main() -> int:
    hr("1. 服务可用性")
    if not check_service():
        if not start_service():
            print(f"{FAIL} 服务启动失败，查看 logs/preflight.err.log")
            return 1
    else:
        print(f"{PASS} 服务已在运行")
    ready = httpx.get(f"{BASE}/ready", timeout=20).json()
    print(f"  数据库: {ready['database']}   向量库: {ready['vector_store']}")
    print(f"  Embedding: {ready['embedding_model']}   重排: {ready['rerank_enabled']}")

    hr("2. 预热（避免首个提问吃冷启动）")
    t0 = time.perf_counter()
    w = httpx.get(f"{BASE}/warm", timeout=300).json()
    print(f"  总耗时 {w.get('total_ms')}ms   ready={w.get('ready')}")
    for k in ("embedding", "rerank", "retrieval"):
        v = w.get(k) or {}
        flag = PASS if v.get("ok") else FAIL
        print(f"  {flag}{k:<10} {v}")
        if not v.get("ok"):
            problems.append(f"预热失败: {k}")

    hr("3. 三个演示账号登录")
    tokens: dict[str, str] = {}
    for u, p, desc in [("admin", "admin123", "管理员(全权限)"),
                       ("zhangsan", "pass1234", "张三 研发/密级2"),
                       ("lisi", "pass1234", "李四 财务/密级3")]:
        try:
            r = httpx.post(f"{BASE}/api/v1/auth/login",
                           json={"tenant": "acme", "username": u, "password": p}, timeout=30)
            r.raise_for_status()
            tokens[u] = r.json()["access_token"]
            print(f"  {PASS}{desc:<20} OK")
        except Exception as e:
            print(f"  {FAIL}{desc:<20} {e}")
            problems.append(f"登录失败: {u}")

    if len(tokens) < 3:
        hr("结论")
        print(f"{FAIL} 账号不全，无法继续")
        return 1

    def hdr(u: str) -> dict:
        return {"Authorization": f"Bearer {tokens[u]}"}

    hr("4. 权限隔离（演示的核心，必须成立）")
    try:
        docs_z = httpx.get(f"{BASE}/api/v1/documents", headers=hdr("zhangsan"), timeout=30).json()
        docs_l = httpx.get(f"{BASE}/api/v1/documents", headers=hdr("lisi"), timeout=30).json()
        tz = {d["title"] for d in docs_z["items"]}
        tl = {d["title"] for d in docs_l["items"]}
        print(f"  张三可见 {len(tz)} 篇: {sorted(tz)}")
        print(f"  李四可见 {len(tl)} 篇: {sorted(tl)}")
        fin = "2024年度财务预算与费用审批权限规定"
        if fin in tz:
            print(f"  {FAIL} 张三居然能看到财务文档（权限泄漏！）")
            problems.append("权限隔离失效")
        else:
            print(f"  {PASS}张三看不到财务文档")
        if fin in tl:
            print(f"  {PASS}李四能看到财务文档")
        else:
            print(f"  {WARN}李四也看不到财务文档（演示效果会打折）")
    except Exception as e:
        print(f"  {FAIL}权限检查异常: {e}")
        problems.append("权限检查异常")

    hr("5. 三类典型问题")
    cases = [
        ("zhangsan", "年假最多可以结转几天？", False, "事实题"),
        ("zhangsan", "50万元以上的费用需要谁审批？", False, "无权题（应拒答）"),
        ("lisi", "50万元以上的费用需要谁审批？", False, "有权题（应作答）"),
        ("zhangsan", "帮我写一首关于春天的诗", False, "超纲题（应拒答）"),
    ]
    for user, q, use_agent, label in cases:
        try:
            t = time.perf_counter()
            r = httpx.post(f"{BASE}/api/v1/chat/ask", headers=hdr(user), timeout=300,
                           json={"question": q, "use_agent": use_agent, "top_k": 5})
            cost = (time.perf_counter() - t) * 1000
            d = r.json()
            flag = PASS if r.status_code == 200 else FAIL
            print(f"  {flag}[{label}] {user}  {cost:.0f}ms  拒答={d.get('refused')} "
                  f"引用={len(d.get('citations') or [])}条")
            print(f"        A: {(d.get('answer') or '')[:80]}")
            if r.status_code != 200:
                problems.append(f"提问失败: {label}")
        except Exception as e:
            print(f"  {FAIL}[{label}] 异常: {e}")
            problems.append(f"提问异常: {label}")

    hr("结论")
    if problems:
        print(f"{FAIL} 发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"   - {p}")
        return 1
    print(f"{PASS}READY FOR DEMO")
    print("\n演示入口：http://127.0.0.1:8000/docs")
    print("演示账号：admin/admin123、zhangsan/pass1234、lisi/pass1234（租户都是 acme）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
