"""
主管智能体 — 多智能体协调的核心。
Supervisor 本身是一个 ReActAgent，通过 delegate_to_agent 工具调用子智能体。
所有路由、规划、合成决策均由 LLM 自主完成，无硬编码规则。
"""

from __future__ import annotations

from app.agents.react import ReActAgent, AgentResult
from app.llm.client import LLMClient
from app.tools import get_tool_registry, ToolRegistry
from app.tools.delegation import get_agent_descriptions


SUPERVISOR_PROMPT = """\
你是一个校园智能助手的主管智能体（Supervisor Agent）。

你的职责：
1. 理解用户的校园相关问题
2. 分析问题，决定是否需要将任务拆解并委托给专业子智能体
3. 汇总各子智能体的结果，给出完整、准确的回答
4. 确保引用了知识库中的相关信息

工作方式：
- 对于简单问题（如查询时间、简单知识检索），使用 search_knowledge_base 等工具直接回答
- 对于复杂问题或需要专业领域知识的问题，使用 delegate_to_agent 工具委托给对应的子智能体
- 收到子智能体结果后，综合整理成用户易于理解的回答

可用工具包括搜索知识库、查询校园指南、获取时间、计算绩点等。
当问题涉及多个专业领域时，可以依次委托给多个子智能体，然后综合结果。

回答要求：
1. 清晰直接地回答用户问题
2. 引用知识库信息来支撑回答
3. 如果信息不足，诚实告知用户
4. 不需要告诉用户内部的工作流程（如"我调用了xxx工具"）
"""


class Supervisor:
    """
    主管智能体：使用 ReAct 循环驱动，LLM 自主决策路由和规划。
    所有行为（路由、委托、规划、合成）均由 LLM 决定，无硬编码规则。
    """

    def __init__(self, llm_client: LLMClient, use_mock: bool = False):
        self.llm_client = llm_client
        self.registry = get_tool_registry()
        self.use_mock = use_mock

        # 构建 Supervisor 可用的工具列表（包含委托工具）
        self.supervisor_tools = self.registry.openai_schemas()

    async def run(
        self,
        query: str,
        *,
        context: str = "",
        user_profile: str = "",
        max_steps: int = 10,
    ) -> AgentResult:
        """运行主管智能体。"""
        agent = ReActAgent(
            name="supervisor",
            system_prompt=(
                SUPERVISOR_PROMPT + "\n\n" + get_agent_descriptions()
            ),
            llm_client=self.llm_client,
            tool_registry=self.registry,
            tools=self.supervisor_tools,
        )

        full_context = context
        if user_profile:
            full_context = f"{user_profile}\n\n{context}" if context else user_profile

        return await agent.run(
            user_message=query,
            context=full_context,
            max_steps=max_steps,
        )
