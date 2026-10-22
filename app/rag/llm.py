"""
LLM 客户端抽象 —— "热切换对话模型"的落地点。

当前用 DeepSeek（deepseek-chat），走 OpenAI 兼容协议。
换成任何 OpenAI 兼容的服务（通义、月之暗面、硅基流动、本地 vLLM/Ollama），
只改 .env 的 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 三行，代码零改动。

模型名与 base_url 的坑（实测踩到过，值得记住）：
    DeepSeek 的 base_url 必须带 /v1（https://api.deepseek.com/v1），
    不带 /v1 时请求会在建立连接后被重置（ConnectionResetError），
    报错信息完全看不出是这个原因 —— 这类"配置型故障"最耗时间。
"""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None


def get_llm_client():
    """全局单例 AsyncOpenAI 客户端。"""
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
            max_retries=2,
        )
        logger.info(f"LLM 就绪: model={settings.LLM_MODEL} base_url={settings.LLM_BASE_URL}")
    return _client


async def chat(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    普通对话调用。

    json_mode=True 时要求模型返回 JSON（DeepSeek 支持 response_format=json_object），
    这是"让大模型输出稳定结构化结果"的第一道保险；
    第二道保险是 Pydantic 校验（见 classify_intent / 引用校验）。
    只靠提示词说"请返回 JSON"是不够的，真实项目里模型总会偶尔加个前缀解释。
    """
    client = get_llm_client()
    kwargs: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage:
        logger.info(
            f"LLM 调用完成 tokens: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )
    return content


async def chat_json(messages: list[dict[str, str]], default: dict | None = None) -> dict:
    """
    要求返回 JSON 并容错解析。

    容错是必须的：即使开了 json_mode，也可能返回空串或非法 JSON
    （网络截断、模型异常）。这里解析失败就返回 default，
    让上层走"保守分支"（例如意图判定失败就当"需要检索"处理），
    而不是让整个问答 500。
    """
    raw = await chat(messages, json_mode=True, temperature=0.0)
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else (default or {})
    except json.JSONDecodeError:
        logger.warning(f"LLM 返回非法 JSON，已降级: {text[:120]!r}")
        return default or {}


async def check_llm() -> tuple[bool, str, int]:
    """健康检查：返回 (是否可用, 说明, 耗时ms)。"""
    import time

    t0 = time.perf_counter()
    try:
        reply = await chat(
            [{"role": "user", "content": "回复两个字：正常"}], max_tokens=8, temperature=0
        )
        ms = int((time.perf_counter() - t0) * 1000)
        return True, f"模型响应: {reply.strip()[:20]}", ms
    except Exception as e:  # noqa: BLE001
        ms = int((time.perf_counter() - t0) * 1000)
        return False, f"{type(e).__name__}: {str(e)[:150]}", ms
