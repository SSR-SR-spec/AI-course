"""
LLM 客户端模块 — 支持 Mock（模拟）模式和 OpenAI 兼容接口（如 DeepSeek 等）。
增强版：支持 function calling / tool calling。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings


@dataclass
class LLMResult:
    """大模型调用的结果封装。"""
    content: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMToolCallResult:
    """
    包含工具调用的大模型响应。
    当模型决定调用工具时，content 为 None，tool_calls 包含调用信息。
    """
    content: str | None
    tool_calls: list[dict[str, Any]] | None  # [{id, type, function: {name, arguments}}]
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """大模型调用客户端，根据配置自动选择 Mock 模式或 OpenAI 兼容模式。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(
        self,
        *,
        system: str,
        user: str,
    ) -> LLMResult:
        """
        简单文本生成（无工具调用）。
        """
        if self.settings.llm_provider == "mock":
            return LLMResult(
                content=_mock_answer(system=system, user=user),
                raw={"provider": "mock"},
            )
        return await _openai_compatible_complete(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            system=system,
            user=user,
            timeout_s=self.settings.llm_timeout_s,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )

    async def chat_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> LLMToolCallResult:
        """
        支持工具调用的对话生成。
        返回可能包含文本回复或工具调用请求。
        """
        max_tokens = max_tokens or self.settings.llm_max_output_tokens

        if self.settings.llm_provider == "mock":
            return self._mock_chat_with_tools(system, messages, tools)

        return await self._llm_chat_with_tools(
            system=system,
            messages=messages,
            tools=tools or [],
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )

    def _mock_chat_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LLMToolCallResult:
        """
        Mock 模式：模拟 LLM 自主决定调用工具或直接回答。
        第 1 轮总返回 search_knowledge_base 工具调用（模拟 LLM 想查知识库），
        后续轮次返回文本回复（模拟 LLM 已有足够信息）。
        不包含任何领域关键词硬编码。
        """
        # 检查是否已有工具调用结果（判断是否在循环中）
        has_tool_results = any(m.get("role") == "tool" for m in messages)

        if tools and not has_tool_results:
            # 第一轮：模拟 LLM 决定调用 search_knowledge_base
            last_user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break
            return LLMToolCallResult(
                content=None,
                tool_calls=[{
                    "id": "mock_tc_1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": json.dumps({"query": last_user_msg[:80], "top_k": 3}, ensure_ascii=False),
                    },
                }],
                raw={"provider": "mock"},
            )

        # 已有工具结果或没有工具：模拟 LLM 给出最终回答
        return LLMToolCallResult(
            content=_mock_answer(system=system, user=messages[-1].get("content", "")),
            tool_calls=None,
            raw={"provider": "mock"},
        )

    async def _llm_chat_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        max_tokens: int,
    ) -> LLMToolCallResult:
        """真实 LLM 工具调用。"""
        from litellm import acompletion

        if not self.settings.openai_api_key:
            raise ValueError("使用 openai_compatible 模式时必须配置 OPENAI_API_KEY。")

        litellm_model = _normalize_litellm_model(
            model=self.settings.openai_model,
            api_base=self.settings.openai_base_url,
        )

        full_messages = list(messages)
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        try:
            kwargs = {
                "model": litellm_model,
                "api_key": self.settings.openai_api_key,
                "api_base": self.settings.openai_base_url,
                "messages": full_messages,
                "temperature": 0.2,
                "timeout": self.settings.llm_timeout_s,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            resp = await acompletion(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"大模型请求失败。原始错误：{type(e).__name__}: {e}"
            ) from e

        return _extract_tool_call_result(resp)


def _mock_answer(*, system: str, user: str) -> str:
    """
    模拟回答：返回固定格式文本。
    用于无需外部 API 即可本地验证 RAG 流程和工作流。
    """
    return (
        "（Mock 模型）\n"
        "我会基于提供的引用片段给出结论；若引用不足，我会提示需要人工确认。\n\n"
        "说明：Mock 模式不会调用外部大模型，因此不会真正'理解'文档内容；"
        "接入 DeepSeek 后才会基于引用片段生成摘要式回答。\n"
    )


async def _openai_compatible_complete(
    *,
    api_key: str | None,
    base_url: str | None,
    model: str,
    system: str,
    user: str,
    timeout_s: int,
    max_output_tokens: int,
) -> LLMResult:
    """使用 litellm 库调用 OpenAI 兼容接口（支持 DeepSeek、OpenAI 等）。"""
    from litellm import acompletion

    if not api_key:
        raise ValueError("使用 openai_compatible 模式时必须配置 OPENAI_API_KEY。")

    try:
        litellm_model = _normalize_litellm_model(model=model, api_base=base_url)
        resp = await acompletion(
            model=litellm_model,
            api_key=api_key,
            api_base=base_url,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            timeout=timeout_s,
            max_tokens=max_output_tokens,
        )
    except Exception as e:
        raise RuntimeError(
            "大模型请求失败。请检查 openai_base_url、openai_model、网络连接 "
            f"以及 API 密钥有效性。原始错误：{type(e).__name__}: {e}"
        ) from e

    content, tool_calls = _extract_content_and_tool_calls(resp)
    return LLMResult(
        content=content or "",
        raw={"provider": "openai_compatible", "model": litellm_model, "resp": resp},
    )


def _normalize_litellm_model(*, model: str, api_base: str | None) -> str:
    """
    规范化模型名：litellm 对许多模型需要显式 provider 前缀，
    如 "deepseek/deepseek-chat"，用户通常只配了 "deepseek-chat"，需要补全前缀。
    """
    m = (model or "").strip()
    if not m:
        raise ValueError("openai_model 配置为空")

    if "/" in m:
        return m

    base = (api_base or "").lower()
    if "deepseek.com" in base:
        if m in {"deepseek", "deepseek-chat", "deepseek_chat"}:
            return "deepseek/deepseek-chat"
        if m.startswith("deepseek-"):
            return f"deepseek/{m}"

    return m


def _extract_content_and_tool_calls(resp: Any) -> tuple[str | None, list[dict] | None]:
    """
    从 litellm 响应中提取文本内容和工具调用。
    """
    try:
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if not choices:
                raise KeyError("choices 字段缺失")
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            return content, tool_calls
    except Exception:
        pass

    try:
        choices = getattr(resp, "choices", None)
        if choices:
            ch0 = choices[0]
            message = getattr(ch0, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None)
                return content, tool_calls
    except Exception:
        pass

    return None, None


def _extract_tool_call_result(resp: Any) -> LLMToolCallResult:
    """从响应中提取工具调用结果。"""
    content, tool_calls = _extract_content_and_tool_calls(resp)

    # 格式化 tool_calls 为可序列化格式
    formatted_tool_calls = None
    if tool_calls:
        formatted_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                formatted_tool_calls.append(tc)
            else:
                formatted_tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", ""),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                    },
                })

    raw = {}
    if isinstance(resp, dict):
        raw = resp
    else:
        try:
            raw = resp.model_dump() if hasattr(resp, "model_dump") else {"resp": str(resp)}
        except Exception:
            raw = {"resp": str(resp)[:200]}

    return LLMToolCallResult(
        content=content,
        tool_calls=formatted_tool_calls,
        raw=raw,
    )
