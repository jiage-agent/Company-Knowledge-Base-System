"""验证降级链路：embedding 不可用时，问答是否还能正常返回（纯关键词检索）。

面试现场如果网络/额度出问题，这个降级就是"服务不会挂"的保证。
"""
import json
import time

import httpx

BASE = "http://127.0.0.1:8000"


def login(user, pwd):
    r = httpx.post(f"{BASE}/api/v1/auth/login",
                   json={"tenant": "acme", "username": user, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    tok = login("zhangsan", "pass1234")
    h = {"Authorization": f"Bearer {tok}"}

    print("=" * 78)
    print("降级链路验证（embedding 额度耗尽的情况下）")
    print("=" * 78)

    for q in ["年假最多可以结转几天？", "一线城市出差住宿标准是多少钱？"]:
        t0 = time.perf_counter()
        r = httpx.post(f"{BASE}/api/v1/chat/ask", headers=h,
                       json={"question": q, "use_agent": False, "top_k": 5}, timeout=180)
        cost = time.perf_counter() - t0
        print(f"\nQ: {q}")
        print(f"   HTTP {r.status_code}  ({cost:.1f}s)")
        if r.status_code != 200:
            print(f"   失败: {r.text[:300]}")
            continue
        d = r.json()
        print(f"   拒答={d['refused']} 引用={len(d['citations'])}条 置信度={d['confidence']}")
        print(f"   A: {d['answer'][:180].replace(chr(10), ' ')}")
        if d["citations"]:
            for c in d["citations"][:2]:
                print(f"      [{c['index']}] 《{c['title'][:26]}》")

    print()
    print("=" * 78)
    print("结论：即使 embedding 完全不可用，服务仍返回答案（质量下降，但不报错）")
    print("=" * 78)


if __name__ == "__main__":
    main()
