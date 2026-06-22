"""
ReAct 循环引擎 — 真正的 LLM 自主驱动的思考-行动-观察循环。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.llm.client import LLMClient
from app.tools import ToolRegistry

logger = logging.getLogger("campus")


@dataclass
class AgentStep:
    role: str  # "thought" | "tool_call" | "tool_result" | "observation" | "final"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls: int = 0


class ReActAgent:
    """
    ReAct 智能体 — 通过工具调用循环自主完成任务。
    每一轮迭代：
      1. LLM 接收当前对话 + 工具定义
      2. LLM 自主决定：继续调用工具，或给出最终回答
      3. 若调用工具 → 执行 → 结果追加到对话 → 回到 1
      4. 若给出回答 → 返回结果
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        tools: list[dict[str, Any]] | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.registry = tool_registry
        self._tools = tools  # OpenAI schema list; if None, use all from registry

    @property
    def tools(self) -> list[dict[str, Any]]:
        if self._tools is not None:
            return self._tools
        return self.registry.openai_schemas()

    async def run(
        self,
        user_message: str,
        *,
        context: str = "",
        user_profile: str = "",
        max_steps: int = 8,
    ) -> AgentResult:
        """
        运行 ReAct 循环。
        Args:
            user_message: 用户输入
            context: 历史对话上下文
            user_profile: 长期记忆中的用户画像
            max_steps: 最大工具调用步数
        """
        # 构建完整的消息列表
        system = self.system_prompt
        if user_profile:
            system = f"{system}\n\n{user_profile}"

        messages: list[dict[str, Any]] = []
        if context:
            messages.append({"role": "user", "content": f"历史对话上下文：\n{context}\n\n当前用户提问：{user_message}"})
        else:
            messages.append({"role": "user", "content": user_message})

        steps: list[AgentStep] = []
        tool_calls_count = 0
        final_answer = ""

        for iteration in range(max_steps):
            # 调用 LLM
            result = await self.llm_client.chat_with_tools(
                system=system,
                messages=messages,
                tools=self.tools,
            )

            if result.tool_calls:
                tool_calls_count += len(result.tool_calls)
                steps.append(AgentStep(
                    role="thought",
                    content=f"第 {iteration + 1} 轮：调用 {len(result.tool_calls)} 个工具",
                ))

                # 将 assistant 消息（含 tool_calls）追加到对话历史
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content}
                if result.tool_calls:
                    assistant_msg["tool_calls"] = result.tool_calls
                messages.append(assistant_msg)

                # 执行每个工具调用
                for tc in result.tool_calls:
                    tc_id = tc.get("id", "")
                    func = tc.get("function", {})
                    tc_name = func.get("name", "")
                    try:
                        tc_args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tc_args = {}

                    tool_result = await self.registry.execute(tc_id, tc_name, tc_args)

                    steps.append(AgentStep(
                        role="tool_call",
                        content=f"{tc_name}({json.dumps(tc_args, ensure_ascii=False)[:100]})",
                        metadata={"tool": tc_name, "args": tc_args},
                    ))

                    if tool_result.success:
                        steps.append(AgentStep(
                            role="tool_result",
                            content=tool_result.output[:200],
                        ))
                    else:
                        steps.append(AgentStep(
                            role="tool_result",
                            content=f"工具执行失败：{tool_result.error}",
                        ))

                    # 将工具结果追加到对话历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result.output if tool_result.success else f"错误：{tool_result.error}",
                    })
            else:
                # LLM 给出了最终回答
                final_answer = result.content or ""
                steps.append(AgentStep(role="final", content=final_answer[:200]))
                return AgentResult(answer=final_answer, steps=steps, tool_calls=tool_calls_count)

        # 超出最大步数，用最后一次结果或超时提示
        if not final_answer:
            final_answer = "无法在限定步骤内完成您的请求。请尝试简化问题或补充更多信息。"
        return AgentResult(answer=final_answer, steps=steps, tool_calls=tool_calls_count)
