"""
问答接口 —— 项目的门面。

演示路径（面试时按这个顺序点，效果最好）：
    1. POST /auth/login          拿 token（分别用 zhangsan / lisi 登录，两个标签页）
    2. GET  /documents           看权限差异：lisi 能看到财务文档，zhangsan 看不到
    3. POST /chat/ask            提问，看返回里的 citations / critic_passed
    4. 同问题换账号               -> 观察权限导致的答案差异（最有说服力的一步）
    5. POST /admin/eval/run      跑消融实验，拿到 Recall@K / NDCG 对比数字

use_agent 参数的意义：
    true  -> 走 LangGraph 多节点编排（查询理解 -> 规划 -> 混合检索 -> 生成 -> Critic 重试）
    false -> 走朴素的固定流水线
    两条链路都保留，是为了让"Agentic 到底带来了什么"可以被现场对比，而不是空口宣称。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from app.core.logging import get_logger, new_trace_id, trace_id_var
from app.core.permissions import PermissionContext
from app.dependencies import PermDep, SessionDep
from app.models import AuditLog, ChatMessage, ChatSession, MessageRole, utcnow
from app.rag.agent import AgenticRAG, run_naive_rag
from app.schemas import AskRequest, AskResponse, Citation, MessageOut, SessionOut

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["问答"])


async def _get_or_create_session(
    session: AsyncSession, perm: PermissionContext, session_uid: str | None, question: str
) -> ChatSession:
    if session_uid:
        stmt = select(ChatSession).where(
            ChatSession.session_uid == session_uid,
            ChatSession.tenant_id == perm.tenant_id,
        )
        cs = (await session.execute(stmt)).scalars().first()
        if cs is None:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_uid}")
        # 会话 id 虽然是随机串，但也不能假设"猜不到就等于有权限"
        if cs.user_id != perm.user_id:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        return cs

    cs = ChatSession(
        session_uid="s" + new_trace_id() + new_trace_id(),
        tenant_id=perm.tenant_id,
        user_id=perm.user_id or 0,
        title=question[:30],
    )
    session.add(cs)
    await session.commit()
    await session.refresh(cs)
    return cs


async def _load_history(session: AsyncSession, session_id: int, limit: int = 6) -> list[dict[str, str]]:
    """
    读取最近几轮对话，用于多轮指代改写。

    只取最近几条而不是全部：一是控制 token 成本，
    二是较早的上下文对当前问题往往是噪声（多轮改写只需要近期指代信息）。
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [{"role": m.role, "content": m.content[:500]} for m in reversed(rows)]


@router.post("/ask", response_model=AskResponse, summary="提问（Agentic RAG / 朴素 RAG 可切换）")
async def ask(
    payload: AskRequest,
    session: SessionDep,
    perm: PermDep,
):
    trace_id_var.set(new_trace_id())
    cs = await _get_or_create_session(session, perm, payload.session_uid, payload.question)
    history = await _load_history(session, cs.id)

    session.add(ChatMessage(
        session_id=cs.id, tenant_id=perm.tenant_id, role=MessageRole.USER.value,
        content=payload.question, trace_id=trace_id_var.get(),
    ))
    await session.commit()

    if payload.use_agent:
        agent = AgenticRAG(session, perm)
        result = await agent.ask(
            payload.question, session_id=cs.id, history=history,
            top_k=payload.top_k, doc_type=payload.doc_type,
        )
    else:
        # 对照组也做多轮改写，保证"只差 Agentic 决策"这一个变量，
        # 否则对比实验就不公平了（控制变量是实验设计的基本要求）
        question = payload.question
        if history:
            from app.rag.llm import chat
            try:
                ctx = "\n".join(f"{h['role']}: {h['content'][:150]}" for h in history[-4:])
                rewritten = (await chat([{"role": "user", "content":
                    f"对话历史：\n{ctx}\n\n最新问题：{payload.question}\n\n"
                    "把最新问题改写为可独立检索的完整问题，只输出问题本身。"}],
                    temperature=0.0, max_tokens=100)).strip()
                question = rewritten.split("\n")[0].strip() or payload.question
            except Exception:  # noqa: BLE001
                question = payload.question
        result = await run_naive_rag(session, perm, question, top_k=payload.top_k)
        result["rewritten_queries"] = [question]

    session.add(ChatMessage(
        session_id=cs.id, tenant_id=perm.tenant_id, role=MessageRole.ASSISTANT.value,
        content=result["answer"], citations=result.get("citations", []),
        intent=result.get("intent"), retrieval_rounds=result.get("retrieval_rounds", 0),
        confidence=result.get("confidence", 0.0), latency_ms=result.get("latency_ms", 0),
        trace_id=result.get("trace_id"), retrieved_count=result.get("retrieved_count", 0),
    ))
    session.add(AuditLog(
        tenant_id=perm.tenant_id, user_id=perm.user_id, action="query",
        resource=f"session:{cs.session_uid}",
        detail={
            "question": payload.question[:200],
            "use_agent": payload.use_agent,
            "intent": result.get("intent"),
            "rounds": result.get("retrieval_rounds"),
            "refused": result.get("refused"),
            "confidence": result.get("confidence"),
            "cited_docs": [c.get("title") for c in result.get("citations", [])],
        },
        trace_id=result.get("trace_id"),
    ))
    cs.updated_at = utcnow()
    await session.commit()

    return AskResponse(
        session_uid=cs.session_uid,
        question=payload.question,
        answer=result["answer"],
        citations=[Citation(**c) for c in result.get("citations", [])],
        confidence=result.get("confidence", 0.0),
        refused=result.get("refused", False),
        intent=result.get("intent", ""),
        need_retrieval=result.get("need_retrieval", True),
        planned_strategy=result.get("planned_strategy", ""),
        rewritten_queries=result.get("rewritten_queries", []),
        retrieval_rounds=result.get("retrieval_rounds", 0),
        retrieved_count=result.get("retrieved_count", 0),
        critic_passed=result.get("critic_passed", False),
        critic_reason=result.get("critic_reason", ""),
        latency_ms=result.get("latency_ms", 0),
        trace_id=result.get("trace_id", ""),
    )


@router.get("/sessions", response_model=list[SessionOut], summary="我的会话列表")
async def list_sessions(
    session: SessionDep,
    perm: PermDep,
    limit: int = Query(20, ge=1, le=100),
):
    stmt = (
        select(ChatSession)
        .where(ChatSession.tenant_id == perm.tenant_id, ChatSession.user_id == perm.user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    out: list[SessionOut] = []
    for cs in rows:
        cnt = (await session.execute(
            select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == cs.id)
        )).scalar() or 0
        out.append(SessionOut(
            session_uid=cs.session_uid, title=cs.title,
            created_at=cs.created_at, updated_at=cs.updated_at, message_count=cnt,
        ))
    return out


@router.get("/sessions/{session_uid}/messages", response_model=list[MessageOut],
            summary="会话消息记录（含引用与运行轨迹）")
async def list_messages(
    session_uid: str,
    session: SessionDep,
    perm: PermDep,
    limit: int = Query(50, ge=1, le=200),
):
    stmt = select(ChatSession).where(
        ChatSession.session_uid == session_uid, ChatSession.tenant_id == perm.tenant_id
    )
    cs = (await session.execute(stmt)).scalars().first()
    if cs is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if cs.user_id != perm.user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")

    stmt = (
        select(ChatMessage).where(ChatMessage.session_id == cs.id)
        .order_by(ChatMessage.created_at).limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        MessageOut(
            id=m.id, role=m.role, content=m.content, citations=m.citations or [],
            intent=m.intent, confidence=m.confidence, latency_ms=m.latency_ms,
            created_at=m.created_at,
        )
        for m in rows
    ]
