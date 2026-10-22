"""
Agentic RAG 编排（LangGraph 多节点 + 条件路由 + 自我纠错）。

这是本项目与"普通 RAG Demo"最大的区别所在，也是简历第一行技术亮点。

一、普通 RAG 与 Agentic RAG 的本质区别（面试必答）：
    普通 RAG 是固定流水线：检索 -> 拼上下文 -> 生成。无论问题难易，流程完全一样。
    它的问题在于：
      - 闲聊/超纲问题也要检索一遍，浪费延迟和 token；
      - 一次检索不准就直接硬答，容易产生幻觉；
      - 无法处理"多跳"问题（需要先查 A 再用 A 的结论查 B）。
    Agentic RAG 把上述决策显式建模成"节点 + 条件边"：
      - 先判断意图：需不需要检索？（闲聊直接兜底，不检索）
      - 规划检索策略：单跳还是多跳？要不要改写查询？
      - 生成后自检：证据够不够？不够就改写查询重检索（最多 N 轮）
      - 仍不够则明确拒答，而不是编造。
    代价是延迟增加（多几次 LLM 调用），收益是准确率与鲁棒性 ——
    这个 trade-off 要能主动说出来，否则会被认为"为了用框架而用框架"。

二、图的拓扑（与简历描述一致）：
    entry -> query_understanding -> retrieval_planner -> hybrid_retriever
          -> reranker(内含在 retriever 里) -> generator -> critic
          -> 条件路由：
               证据充分        -> END（返回答案 + 引用）
               证据不足且未超轮次 -> 回到 retrieval_planner（换策略/改写查询）
               证据不足且已超轮次 -> 人工兜底 / 明确拒答 -> END

三、为什么要用 LangGraph 而不是自己写 if-else：
    1. 状态显式化：整个流程共享一个 TypedDict 状态，
       "它为什么这么答"可以逐步复现（可解释、可单测）；
    2. 条件边是声明式的：拓扑改了不用改业务流程代码；
    3. 自带递归上限保护（recursion_limit），避免条件循环变成死循环 ——
       自己写 while 循环最容易出的 bug 就是跑不完。
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.core.logging import get_logger, trace_id_var
from app.core.permissions import PermissionContext, RetrievedChunk
from app.models import ChatMessage, ChatSession, Document, MessageRole
from app.rag.generator import (
    REFUSAL_TEXT,
    compute_confidence,
    generate_answer,
    verify_citations,
    verify_numbers,
)
from app.rag.hybrid import HybridRetriever
from app.rag.llm import chat_json

logger = get_logger(__name__)


def merge_chunks(old: list, new: list) -> list:
    """
    状态里的 chunk 列表用"追加"语义。

    为什么必须声明 reducer：LangGraph 默认对状态字段做「覆盖」。
    如果检索重试时是覆盖，第二轮会把第一轮召回到的证据丢掉；
    用追加 + 去重才能实现"多轮检索的证据累积"，这是多跳场景的关键。
    """
    seen = {c.chunk_uid for c in old}
    merged = list(old)
    for c in new:
        if c.chunk_uid not in seen:
            merged.append(c)
            seen.add(c.chunk_uid)
    return merged


class RAGState(TypedDict, total=False):
    """整张图共享的状态。字段即"决策依据"，可完整复现一次问答的内部过程。"""

    # 输入
    question: str
    session_id: int
    tenant_id: int
    top_k: int
    doc_type: str | None
    history: list[dict[str, str]]

    # 查询理解
    normalized_question: str
    intent: str                # factual | comparison | multi_hop | chitchat
    need_retrieval: bool
    rewritten_queries: list[str]

    # 检索
    planned_strategy: str      # single | multi_query | decompose
    retrieval_round: int
    retrieved: Annotated[list[RetrievedChunk], merge_chunks]
    retrieval_debug: list[dict]

    # 生成与校验
    answer: str
    citations: list[dict]
    confidence: float
    refused: bool
    critic_passed: bool
    critic_reason: str

    # 观测
    latency_ms: int
    trace_id: str
    error: str


class AgenticRAG:
    """
    把 LangGraph 图封装成可复用对象。

    注意：session 是"每次请求一个"的东西，不适合放进图状态
    （状态要可序列化，供 checkpointer 持久化）。
    所以用实例属性持有，并用 build_graph() 时闭包捕获 ——
    这是 LangGraph 里"依赖注入"的常见写法。
    """

    def __init__(self, session: AsyncSession, perm: PermissionContext) -> None:
        self.session = session
        self.perm = perm
        self.retriever = HybridRetriever(session)
        self.graph = self._build_graph()

    # ---------------- 节点 1：查询理解 ----------------
    async def node_query_understanding(self, state: RAGState) -> dict:
        """
        做三件事：归一化 -> 意图分类 -> 决定是否检索。

        为什么这一步不能省（价值最高的节点之一）：
          1. "你好"这类闲聊如果走检索，白花一次 embedding + 一次向量查询 + 一次 LLM；
          2. 不同意图需要不同检索策略，先分类才能规划；
          3. 归一化能处理错别字与口语（"年假咋算" -> "年假如何计算"），
             对稀疏检索（关键词匹配）提升尤其明显。
        """
        question = state["question"].strip()
        history = state.get("history") or []

        # 多轮改写：把"那它的有效期呢"这类指代补全成独立问题。
        # 没有这一步，多轮对话的检索质量会断崖式下降（这是最容易被忽略的坑）。
        if history:
            ctx = "\n".join(f"{h['role']}: {h['content'][:200]}" for h in history[-4:])
            prompt = (
                f"以下是对话历史：\n{ctx}\n\n"
                f"用户最新问题：{question}\n\n"
                "如果最新问题包含指代（如\"它\"\"这个\"\"那\"）或省略了主语，"
                "请把它改写成一个可以独立检索的完整问题；"
                "否则原样返回。只输出改写后的问题本身，不要解释。"
            )
            try:
                from app.rag.llm import chat
                rewritten = (await chat([{"role": "user", "content": prompt}],
                                        temperature=0.0, max_tokens=100)).strip()
                if rewritten and len(rewritten) < 300:
                    question = rewritten.split("\n")[0].strip()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"多轮改写失败（降级为原问题）: {e}")

        # 意图分类：让模型输出结构化 JSON，并用 Pydantic 思路做好兜底
        classify_prompt = (
            "判断用户问题的类型，输出 JSON。\n"
            "intent 取值：factual(事实查询) / comparison(对比) / multi_hop(多跳推理) / chitchat(闲聊或与知识库无关)\n"
            "need_retrieval：是否需要查企业知识库\n\n"
            '输出格式：{"intent":"factual","need_retrieval":true}\n\n'
            f"用户问题：{question}"
        )
        try:
            data = await chat_json([{"role": "user", "content": classify_prompt}],
                                   default={"intent": "factual", "need_retrieval": True})
            intent = str(data.get("intent", "factual"))
            need = bool(data.get("need_retrieval", True))
        except Exception as e:  # noqa: BLE001
            # 分类失败时走保守分支：当作需要检索的事实问题。
            # 宁可多检索一次，也不要把该检索的问题当成闲聊直接拒答。
            logger.warning(f"意图分类失败，降级为需要检索: {e}")
            intent, need = "factual", True

        logger.info(f"[查询理解] intent={intent} need_retrieval={need} 改写后={question!r}")
        return {
            "normalized_question": question,
            "intent": intent,
            "need_retrieval": need,
        }

    # ---------------- 节点 2：检索规划 ----------------
    async def node_retrieval_planner(self, state: RAGState) -> dict:
        """
        决定"怎么检索"：单跳 / 多查询 / 查询分解。

        策略与意图的对应关系（有依据，不是拍脑袋）：
          factual     -> single：单一事实查询，一次检索足够
          comparison  -> decompose：对比问题（"A 和 B 哪个更优"）需要分别检索 A 和 B 的证据
          multi_hop   -> multi_query：多跳问题需要先查中间实体，再查目标
        重试轮次（retrieval_round > 0）时强制换策略：
          这是防止"重检索空转"的关键 —— 如果第二轮还用同样的查询，
          必然召回同一批结果，重试就没意义了。
        """
        round_no = state.get("retrieval_round", 0)
        intent = state.get("intent", "factual")
        question = state.get("normalized_question") or state["question"]

        if round_no == 0:
            strategy = {"comparison": "decompose", "multi_hop": "multi_query"}.get(intent, "single")
            queries = [question]
        else:
            # 重试：换一种召回视角。用"从证据中提炼关键词"的查询，
            # 比原问句更可能命中之前没召回到的块（避免同样的查询同样的结果）。
            strategy = "multi_query"
            queries = await self._rewrite_for_retry(question, state.get("retrieved", []))

        logger.info(f"[检索规划] 轮次={round_no} 策略={strategy} 查询数={len(queries)}")
        return {"planned_strategy": strategy, "rewritten_queries": queries}

    async def _rewrite_for_retry(self, question: str, got: list[RetrievedChunk]) -> list[str]:
        """
        重检索时的查询改写。

        ⚠️ 这里刻意【不做 HyDE】（重要设计决策，面试可讲）：
            HyDE 的思路是先让模型"编一段假想答案"再拿它去检索。
            但企业知识库场景下这非常危险：模型会编出原文里根本没有的数字
            （实测中它把"住宿标准"编成了 500 元 / 董事长审批），
            这些幻觉词进入查询后会把检索彻底带偏，召回的全是不相关块。
            所以这里的改写只允许使用**问题中已有的词**：
              1. 关键词抽取：把问题里的实体、数字、英文编号单独成查询；
              2. 同义改写：换一种问法（只换措辞，不引入新事实）；
              3. 子问题拆分：把复合问题拆成两个更简单的查询。
            宁可召回少一点，也不能让幻觉污染检索 —— 这是"可控性优先"的取舍。

        为什么重试必须换查询（而不是原查询再跑一遍）：
            向量检索是确定性的，同样的查询必然返回同样的结果，
            "重试"就成了纯粹的延迟浪费。所以第二轮必须换一个召回视角。
        """
        keywords = self._extract_keywords(question)
        seen_titles = "、".join({c.title for c in got[:3] if c.title}) or "无"

        prompt = (
            f"用户问题：{question}\n"
            f"已检索到的资料来源：{seen_titles}（但未能充分回答问题）\n\n"
            "请生成 3 个不同的检索查询，帮助找到答案。严格约束：\n"
            "1. 只能使用问题中已出现的词汇、数字、术语，禁止添加任何问题中没有的信息；\n"
            "2. 禁止引入统计与评价类词汇（标准差、差额、平均值、多少倍、优劣、对比结论），\n"
            "   这些词在制度文档里根本不会出现，只会把检索带偏；\n"
            "3. 三个查询分别从「关键词组合」「同义换词」「拆分子问题」三个角度出发；\n"
            "4. 每个查询都是独立的检索语句，不要写成一句话解释。\n"
            '只输出 JSON：{"queries":["...","...","..."]}'
        )
        queries: list[str] = []
        try:
            data = await chat_json([{"role": "user", "content": prompt}], default={})
            queries = [q.strip() for q in (data.get("queries") or [])
                       if isinstance(q, str) and q.strip()]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"查询改写失败: {e}")

        # 规则兜底：模型改写不可用时，用关键词抽取 + 子问题拆分顶上。
        # 有兜底方案是"重试机制真的能工作"的保证，否则模型一抖动重试就形同虚设。
        if not queries:
            queries = keywords
        else:
            # 把规则抽取的关键词也加进去：它能覆盖模型可能忽略的数字/编号，
            # 而数字与编号恰恰是 BM25 最擅长、也最容易决定成败的部分
            queries = queries[:2] + keywords[:1]
        return queries[:3] or [question]
    @staticmethod
    def _extract_keywords(question: str, max_terms: int = 4) -> list[str]:
        """
        规则化关键词抽取（不依赖 LLM）。

        抽取顺序按"信息量"排：
          英文/数字编号（INC-2024-0912、P0、800）> 中文长词 > 原问题兜底。
        为什么优先编号与数字：
            这类 token 在 embedding 空间里区分度低（训练语料中罕见），
            向量检索常常抓不住，而用户往往恰恰是在问这些东西。
            把它们单独拎出来做一路 BM25 查询，能显著提升召回。
        """
        import re

        terms: list[str] = []
        # 1. 英文/数字/连字符编号
        terms += re.findall(r"[A-Za-z]+-\d{4}-\d+", question)
        terms += re.findall(r"\b[A-Z]{1,3}\d{1,3}\b", question)
        terms += re.findall(r"\d+(?:\.\d+)?", question)
        # 2. 中文实词：取连续汉字串，再切成 2~3 字片段（近似分词）
        for seg in re.findall(r"[\u4e00-\u9fff]{2,}", question):
            for size in (3, 2):
                for i in range(len(seg) - size + 1):
                    terms.append(seg[i:i + size])
        # 去重、去停用词式短词、按出现顺序保留
        # 停用词里包含"统计与提问类虚词"：
        #   问句里天然带"多少/怎么/标准差/差额"这类词，它们在制度文档中几乎不出现，
        #   留作检索词只会命中噪声（实测"标准差"把住宿标准条款的排名压到了 0.24）。
        stop = {"多少", "怎么", "如何", "什么", "哪一", "是否", "可以", "需要", "我们", "公司",
                "标准", "差多少", "标准差", "差额", "平均", "区别", "不同", "分别", "以及",
                "一个", "这个", "那个", "有没有", "请问", "告诉我"}
        uniq: list[str] = []
        for t in terms:
            if t in stop or t in uniq or len(t) < 2:
                continue
            uniq.append(t)
        # 组合成 1~2 条查询：
        #   一条短查询（2~3 个词）命中率高，适合 BM25 的精确匹配；
        #   一条稍长的把更多线索放在一起，适合向量检索的语义匹配。
        if not uniq:
            return [question]
        short = " ".join(uniq[:3])
        long_ = " ".join(uniq[:8])
        return [short, long_] if short != long_ else [short]

    # ---------------- 节点 3：混合检索 ----------------
    async def node_hybrid_retriever(self, state: RAGState) -> dict:
        """
        执行混合检索（并行扇出多条查询）。

        扇出（fan-out）为什么重要：
          多跳/对比类问题拆成多个子查询后，如果串行检索，
          延迟会线性叠加（3 个查询 = 3 倍等待）。
          用 asyncio.gather 并行，总延迟接近单次检索 ——
          这是"多跳能力"与"响应速度"能同时成立的唯一办法。
        """
        queries = state.get("rewritten_queries") or [state["question"]]
        top_k = state.get("top_k") or settings.RERANK_TOP_K

        where_extra = None
        if state.get("doc_type"):
            where_extra = {"doc_type": {"$eq": state["doc_type"]}}

        import asyncio

        tasks = [
            self.retriever.retrieve(
                q, self.perm, top_k=top_k, where_extra=where_extra
            )
            for q in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks: list[RetrievedChunk] = []
        debug_list: list[dict] = []
        for q, res in zip(queries, results):
            if isinstance(res, Exception):
                logger.error(f"检索失败 query={q!r}: {res}")
                continue
            chunks, dbg = res
            all_chunks.extend(chunks)
            dbg["query"] = q
            debug_list.append(dbg)

        # 多查询结果按分数重排 + 去重（同一块可能被多个子查询命中）
        uniq: dict[str, RetrievedChunk] = {}
        for c in all_chunks:
            if c.chunk_uid not in uniq or c.score > uniq[c.chunk_uid].score:
                uniq[c.chunk_uid] = c
        merged = sorted(uniq.values(), key=lambda x: x.score, reverse=True)[:top_k]

        logger.info(f"[混合检索] 轮次={state.get('retrieval_round', 0)} "
                    f"查询={len(queries)} 去重后={len(merged)}")
        return {
            "retrieved": merged,
            "retrieval_debug": debug_list,
            "retrieval_round": state.get("retrieval_round", 0) + 1,
        }

    # ---------------- 节点 4：生成 ----------------
    async def node_generator(self, state: RAGState) -> dict:
        chunks = state.get("retrieved", [])
        question = state.get("normalized_question") or state["question"]
        result = await generate_answer(question, chunks, state.get("history"))

        citations = [c.to_citation(i + 1) for i, c in enumerate(result["used_chunks"])]
        logger.info(f"[生成] 引用数={len(citations)} 拒答={result['refused']} "
                    f"上下文字符={result['prompt_chars']}")
        return {
            "answer": result["answer"],
            "citations": citations,
            "refused": result["refused"],
        }

    # ---------------- 节点 5：Critic 校验 ----------------
    async def node_critic(self, state: RAGState) -> dict:
        """
        质检：答案是否被证据支持。

        三重校验（从便宜到贵，能便宜解决就不叫模型）：
          1. 规则：引用编号是否真实存在（编造引用 = 幻觉的直接证据）；
          2. 规则：答案中的数字是否都能在证据里找到（数字错误最致命）；
          3. LLM：整体是否被证据支持（兜住前两条抓不到的语义层面偏差）。

        为什么先规则后模型：
          规则校验零成本、确定性、可解释；模型校验有成本、有随机性。
          先用规则拦掉大部分问题，再让模型处理剩下需要"理解"的情况，
          这是成本与可靠性的双重优化。
        """
        chunks = state.get("retrieved", [])
        answer = state.get("answer", "")
        used_idx = {i + 1 for i in range(len(state.get("citations", [])))}

        if state.get("refused") or not chunks:
            return {"critic_passed": False, "critic_reason": "无检索结果或已拒答"}

        # 规则 1：引用真实性
        cite_check = verify_citations(answer, chunks)
        # 规则 2：数字一致性
        num_check = verify_numbers(answer, chunks)
        confidence = compute_confidence(chunks, cite_check, num_check)

        reason_parts = []
        if not cite_check["has_citation"]:
            reason_parts.append("答案没有任何引用标注")
        if cite_check["invalid"]:
            reason_parts.append(f"引用了不存在的来源编号 {cite_check['invalid']}")
        if not num_check["ok"]:
            reason_parts.append(f"答案中的数字在证据里找不到依据: {num_check['unsupported']}")

        if reason_parts:
            return {
                "critic_passed": False,
                "critic_reason": "；".join(reason_parts),
                "confidence": confidence,
            }

        # 规则 3：LLM 判定（只在规则通过后才调用，省成本）
        evidence = "\n".join(f"[{i+1}] {(c.parent_content or c.content)[:500]}"
                             for i, c in enumerate(chunks))
        judge_prompt = (
            "你是严格的事实核查员。判断【答案】是否完全被【证据】支持。\n"
            "只要答案包含证据中没有的信息（即使看起来很合理），就判为不支持。\n"
            '输出 JSON：{"supported":true/false,"reason":"简短理由"}\n\n'
            f"【证据】\n{evidence}\n\n【答案】\n{answer}"
        )
        try:
            data = await chat_json([{"role": "user", "content": judge_prompt}],
                                   default={"supported": True, "reason": "校验不可用，默认通过"})
            supported = bool(data.get("supported", True))
            reason = str(data.get("reason", ""))[:200]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM 校验失败，降级为规则结论: {e}")
            supported, reason = True, "LLM 校验不可用"

        return {
            "critic_passed": supported,
            "critic_reason": reason,
            "confidence": confidence if supported else round(confidence * 0.6, 4),
        }

    # ---------------- 条件路由 ----------------
    def route_after_critic(self, state: RAGState) -> str:
        """
        决定下一步走向。这是"Agentic"的核心：流程由运行时状态决定，而不是写死。

        三条分支：
          通过                    -> end
          不通过且还有重试额度      -> retry（回到 planner，换策略重检索）
          不通过且额度用尽          -> give_up（拒答/转人工，绝不硬答）
        """
        if state.get("critic_passed"):
            return "end"
        attempt = state.get("retrieval_round", 0)
        if attempt <= settings.MAX_RETRY and state.get("retrieved"):
            logger.info(f"[路由] 证据不足，进入第 {attempt + 1} 轮检索"
                        f"（原因：{state.get('critic_reason', '')[:60]}）")
            return "retry"
        logger.warning(f"[路由] 重试额度用尽，拒答（原因：{state.get('critic_reason', '')[:60]}）")
        return "give_up"

    def node_give_up(self, state: RAGState) -> dict:
        """兜底：明确拒答，而不是编一个答案。"""
        return {
            "answer": REFUSAL_TEXT + (
                f"（已尝试 {state.get('retrieval_round', 0)} 轮检索，"
                f"原因：{state.get('critic_reason', '证据不足')}）"
            ),
            "refused": True,
            "confidence": min(state.get("confidence", 0.0), 0.3),
        }

    def node_chitchat(self, state: RAGState) -> dict:
        """
        闲聊/超纲问题的兜底。

        安全要点（企业场景必须这么处理）：
          这里【不能】放行给通用大模型自由回答。
          员工问"公司明年战略是什么"，模型若自由发挥就是合规事故。
          所以只回固定话术，并明确告知"未使用知识库"。
        """
        return {
            "answer": "您好，我是企业知识库问答助手，只能回答公司内部文档相关的问题。"
                      "请把问题聚焦到制度、流程、规范等公司文档内容上。",
            "citations": [],
            "refused": True,
            "confidence": 1.0,
            "critic_passed": True,
        }

    # ---------------- 组装图 ----------------
    def _build_graph(self):
        g = StateGraph(RAGState)

        g.add_node("query_understanding", self.node_query_understanding)
        g.add_node("retrieval_planner", self.node_retrieval_planner)
        g.add_node("hybrid_retriever", self.node_hybrid_retriever)
        g.add_node("generator", self.node_generator)
        g.add_node("critic", self.node_critic)
        g.add_node("give_up", self.node_give_up)
        g.add_node("chitchat", self.node_chitchat)

        g.set_entry_point("query_understanding")

        # 条件边：闲聊直接走兜底，不检索（省一次完整检索链路的成本）
        g.add_conditional_edges(
            "query_understanding",
            lambda s: "chitchat" if not s.get("need_retrieval", True) else "retrieval_planner",
            {"chitchat": "chitchat", "retrieval_planner": "retrieval_planner"},
        )
        g.add_edge("retrieval_planner", "hybrid_retriever")
        g.add_edge("hybrid_retriever", "generator")
        g.add_edge("generator", "critic")

        # Critic 之后的三条分支（含回环）
        g.add_conditional_edges(
            "critic",
            self.route_after_critic,
            {"end": END, "retry": "retrieval_planner", "give_up": "give_up"},
        )
        g.add_edge("give_up", END)
        g.add_edge("chitchat", END)

        return g.compile()

    # ---------------- 对外入口 ----------------
    async def ask(
        self,
        question: str,
        session_id: int,
        history: list[dict[str, str]] | None = None,
        top_k: int = 5,
        doc_type: str | None = None,
    ) -> dict:
        """执行一次完整问答，并把过程与结果落库（消息表 + 审计）。"""
        t0 = time.perf_counter()
        trace_id = trace_id_var.get()

        init_state: RAGState = {
            "question": question,
            "session_id": session_id,
            "tenant_id": self.perm.tenant_id,
            "top_k": top_k,
            "doc_type": doc_type,
            "history": history or [],
            "retrieval_round": 0,
            "retrieved": [],
            "retrieval_debug": [],
            "rewritten_queries": [],
            "refused": False,
            "critic_passed": True,
            "trace_id": trace_id,
        }

        final_state = await self.graph.ainvoke(
            init_state,
            config={"recursion_limit": settings.GRAPH_RECURSION_LIMIT},
        )

        latency = int((time.perf_counter() - t0) * 1000)
        return {
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "confidence": final_state.get("confidence", 0.0),
            "refused": final_state.get("refused", False),
            "intent": final_state.get("intent", ""),
            "need_retrieval": final_state.get("need_retrieval", True),
            "planned_strategy": final_state.get("planned_strategy", ""),
            "rewritten_queries": final_state.get("rewritten_queries", []),
            "retrieval_rounds": final_state.get("retrieval_round", 0),
            "retrieved_count": len(final_state.get("retrieved", [])),
            "critic_passed": final_state.get("critic_passed", False),
            "critic_reason": final_state.get("critic_reason", ""),
            "latency_ms": latency,
            "trace_id": trace_id,
            "retrieval_debug": final_state.get("retrieval_debug", []),
        }


async def run_naive_rag(
    session: AsyncSession, perm: PermissionContext, question: str, top_k: int = 5
) -> dict:
    """
    朴素 RAG（单链、无 Agent 决策）—— 作为 Agentic 的对照组。

    为什么专门实现一个"对照组"（这是能加分的细节）：
      只说"我用了 Agentic RAG"没有说服力，
      要能回答"它比普通 RAG 好多少"。有了这个函数，
      评测脚本就能跑出"A 组 vs B 组"的对比数字，
      而且它也是演示时的绝佳素材：同一次提问，两条链路结果不同。
    """
    t0 = time.perf_counter()
    retriever = HybridRetriever(session)
    chunks, _ = await retriever.retrieve(question, perm, top_k=top_k, mode="dense_bm25_rerank")
    result = await generate_answer(question, chunks)
    cite_check = verify_citations(result["answer"], result["used_chunks"])
    num_check = verify_numbers(result["answer"], result["used_chunks"])
    return {
        "answer": result["answer"],
        "citations": [c.to_citation(i + 1) for i, c in enumerate(result["used_chunks"])],
        "confidence": compute_confidence(chunks, cite_check, num_check),
        "refused": result["refused"],
        "intent": "n/a",
        "need_retrieval": True,
        "planned_strategy": "fixed-pipeline",
        "rewritten_queries": [question],
        "retrieval_rounds": 1,
        "retrieved_count": len(chunks),
        "critic_passed": True,
        "critic_reason": "未启用校验",
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "trace_id": trace_id_var.get(),
        "retrieval_debug": [],
    }
