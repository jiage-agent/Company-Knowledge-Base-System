"""端到端接口测试：登录 -> 文档列表 -> 提问 -> 权限隔离验证。

把"演示脚本"和"测试脚本"合一，好处是：每次改代码都跑一遍，
既验证功能没坏，也保证演示流程可用（演示脚本最容易因为改动而失效）。
"""

import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
OK = "[OK]  "
FAIL = "[FAIL]"


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures = 0
    client = httpx.Client(timeout=180.0)

    # ---------- 1. 健康检查 ----------
    hr("1. 就绪探针 /ready")
    r = client.get(f"{BASE}/ready")
    print(f"  HTTP {r.status_code}")
    print("  " + json.dumps(r.json(), ensure_ascii=False, indent=2).replace("\n", "\n  "))

    # ---------- 2. 登录 ----------
    hr("2. 登录三个账号（不同权限视角）")
    tokens = {}
    for user, pwd, desc in [
        ("admin", "admin123", "管理员 admin（全权限）"),
        ("zhangsan", "pass1234", "张三 研发部 / 密级2"),
        ("lisi", "pass1234", "李四 财务部 / 密级3"),
    ]:
        r = client.post(f"{BASE}/api/v1/auth/login",
                        json={"tenant": "acme", "username": user, "password": pwd})
        if r.status_code == 200:
            tokens[user] = r.json()["access_token"]
            me = client.get(f"{BASE}/api/v1/auth/me",
                            headers={"Authorization": f"Bearer {tokens[user]}"}).json()
            print(f"  {OK} {desc:<26} -> role={me['role']} dept={me['department']} "
                  f"level={me['security_level']}")
        else:
            print(f"  {FAIL} {user}: HTTP {r.status_code} {r.text[:120]}")
            failures += 1

    if len(tokens) < 3:
        print("\n登录失败，终止测试")
        return 1

    # ---------- 3. 文档列表（权限差异）----------
    hr("3. 文档列表 —— 观察不同账号能看到的文档不同（权限下推生效）")
    for user in ["admin", "zhangsan", "lisi"]:
        r = client.get(f"{BASE}/api/v1/documents",
                       headers={"Authorization": f"Bearer {tokens[user]}"})
        data = r.json()
        titles = [f"{d['title'][:16]}(密级{d['security_level']}, {d['department'] or '全公司'})"
                  for d in data["items"]]
        print(f"  {user:<9} 可见 {len(titles)} 份: {titles}")

    # ---------- 4. 组件健康 ----------
    hr("4. 各依赖组件健康检查（演示依赖故障时如何降级的素材）")
    r = client.get(f"{BASE}/api/v1/admin/health/components",
                   headers={"Authorization": f"Bearer {tokens['admin']}"})
    for c in r.json():
        flag = OK if c["ok"] else FAIL
        print(f"  {flag} {c['name']:<32} {c['latency_ms']:>5}ms  {c['detail'][:60]}")

    # ---------- 5. 提问 ----------
    hr("5. Agentic RAG 提问（含意图识别、检索轮次、引用、Critic 校验）")
    asks = [
        ("zhangsan", "司龄满3年不满5年的员工年假有多少天？"),
        ("zhangsan", "一线城市和二线城市的住宿标准差多少？"),
        ("zhangsan", "如果一次业务招待花费3000元，需要谁审批？"),
        ("zhangsan", "公司明年的战略规划是什么？"),
    ]
    for user, q in asks:
        t0 = time.perf_counter()
        r = client.post(f"{BASE}/api/v1/chat/ask",
                        headers={"Authorization": f"Bearer {tokens[user]}"},
                        json={"question": q, "use_agent": True, "top_k": 5})
        cost = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"  {FAIL} {q} -> HTTP {r.status_code} {r.text[:200]}")
            failures += 1
            continue
        d = r.json()
        print(f"\n  Q: {q}")
        print(f"     意图={d['intent']} 策略={d['planned_strategy']} "
              f"检索轮次={d['retrieval_rounds']} 命中={d['retrieved_count']}条 "
              f"置信度={d['confidence']} Critic={'通过' if d['critic_passed'] else '未通过'} "
              f"({cost:.1f}s)")
        if d["rewritten_queries"] and d["rewritten_queries"][0] != q:
            print(f"     改写查询: {d['rewritten_queries'][0][:60]}")
        print(f"     A: {d['answer'][:220].replace(chr(10), ' ')}")
        if d["citations"]:
            for c in d["citations"][:3]:
                print(f"        [{c['index']}] 《{c['title'][:20]}》 第{c['page_no']}页 "
                      f"score={c['score']:.4f}")
        if d["refused"]:
            print(f"     -> 拒答（未编造）")
        if not d["critic_passed"]:
            print(f"     Critic 原因: {d['critic_reason'][:100]}")

    # ---------- 6. 权限隔离关键验证 ----------
    hr("6. 【核心演示】同一问题、不同账号 -> 权限隔离是否真的生效")
    q = "50万元以上的费用需要谁审批？"
    for user in ["zhangsan", "lisi"]:
        r = client.post(f"{BASE}/api/v1/chat/ask",
                        headers={"Authorization": f"Bearer {tokens[user]}"},
                        json={"question": q, "use_agent": False, "top_k": 5})
        d = r.json()
        print(f"\n  {user}（{ '研发/密级2' if user=='zhangsan' else '财务/密级3' }）提问：{q}")
        print(f"     引用数={len(d['citations'])} 拒答={d['refused']} 置信度={d['confidence']}")
        print(f"     答案: {d['answer'][:200].replace(chr(10),' ')}")
        if d["citations"]:
            print(f"     引用文档: {[c['title'][:18] for c in d['citations']]}")

    # ---------- 7. 多轮对话 ----------
    hr("7. 多轮对话（演示指代消解：第二句里的指代能否被正确理解）")
    r = client.post(f"{BASE}/api/v1/chat/ask",
                    headers={"Authorization": f"Bearer {tokens['zhangsan']}"},
                    json={"question": "年假最多可以结转几天？", "use_agent": True})
    d1 = r.json()
    print(f"  第1轮 Q: 年假最多可以结转几天？")
    print(f"        A: {d1['answer'][:150].replace(chr(10),' ')}")
    r = client.post(f"{BASE}/api/v1/chat/ask",
                    headers={"Authorization": f"Bearer {tokens['zhangsan']}"},
                    json={"question": "那它需要在几个月内用完？", "session_uid": d1["session_uid"],
                          "use_agent": True})
    d2 = r.json()
    print("  第2轮 Q: 那它需要在几个月内用完？（含指代）")
    print(f"        改写后查询: {d2['rewritten_queries'][0][:70] if d2['rewritten_queries'] else '无'}")
    print(f"        A: {d2['answer'][:200].replace(chr(10),' ')}")

    # ---------- 8. 消融实验（小样本）----------
    hr("8. 消融实验（小样本 6 条，验证评测链路可用；完整版用 scripts/eval.py 跑）")
    r = client.post(f"{BASE}/api/v1/admin/eval/run",
                    headers={"Authorization": f"Bearer {tokens['admin']}"},
                    json={"modes": ["dense_only", "dense_bm25", "dense_bm25_rerank"],
                          "top_k": 5, "limit": 6})
    if r.status_code == 200:
        d = r.json()
        print(f"  run_id={d['run_id']} 样本={d['dataset_size']} 模式数={len(d['metrics'])}")
        print(f"  {'配置':<30} {'Recall@K':>10} {'MRR':>8} {'NDCG':>8} {'HitRate':>9} {'延迟ms':>9}")
        for m in d["metrics"]:
            print(f"  {m['mode'][:28]:<30} {m['recall_at_k']:>10.4f} {m['mrr']:>8.4f} "
                  f"{m['ndcg']:>8.4f} {m['hit_rate']:>9.4f} {m['avg_latency_ms']:>9.1f}")
    else:
        print(f"  {FAIL} HTTP {r.status_code} {r.text[:300]}")
        failures += 1

    # ---------- 9. 审计日志 ----------
    hr("9. 统计信息")
    r = client.get(f"{BASE}/api/v1/admin/stats",
                   headers={"Authorization": f"Bearer {tokens['admin']}"})
    print("  " + json.dumps(r.json(), ensure_ascii=False))

    hr(f"测试结束：失败项 = {failures}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
