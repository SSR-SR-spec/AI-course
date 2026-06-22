"""
Agent 委托工具 — 将任务委托给专业子智能体执行。
子智能体拥有独立的 ReAct 循环和系统提示词。
"""

from __future__ import annotations

from typing import Any

from app.tools import ToolDef, get_tool_registry


AGENT_DEFINITIONS: list[dict[str, str]] = [
    {
        "name": "academic",
        "display": "学术智能体",
        "description": "处理选课、培养方案、学分、绩点、毕业要求、考试安排等学术问题",
        "system_prompt": (
            "你是校园智能助手中的学术智能体（Academic Agent）。\n"
            "你的职责：回答关于课程选择、培养方案、学分要求、绩点计算、毕业条件等方面的学术问题。\n"
            "使用 search_knowledge_base 工具查询知识库中的学术政策文件。\n"
            "使用 calculate_gpa 工具计算绩点。\n"
            "回答要求：\n"
            "1. 优先引用知识库中的具体文件条款\n"
            "2. 如果信息不足，明确告知用户并建议咨询教务部门\n"
            "3. 涉及数据计算时，使用工具精确计算\n"
        ),
    },
    {
        "name": "life",
        "display": "生活智能体",
        "description": "处理食堂、图书馆、校车、宿舍、校园卡、校园设施等校园生活问题",
        "system_prompt": (
            "你是校园智能助手中的生活智能体（Life Agent）。\n"
            "你的职责：回答关于食堂餐饮、图书馆服务、校车安排、宿舍管理、校园设施等生活问题。\n"
            "使用 search_knowledge_base 工具查询知识库中的校园生活信息。\n"
            "使用 get_campus_guide 工具查询校园指南信息。\n"
            "回答要求：\n"
            "1. 提供具体准确的时间、地点和操作指引\n"
            "2. 如果信息不足，告知用户可咨询的部门或电话\n"
        ),
    },
    {
        "name": "admin",
        "display": "行政智能体",
        "description": "处理奖学金、请假、报销、审批流程、资助申请、证明材料等行政事务",
        "system_prompt": (
            "你是校园智能助手中的行政智能体（Admin Agent）。\n"
            "你的职责：回答关于奖学金申请、请假流程、报销手续、审批流程、资助政策等方面的问题。\n"
            "使用 search_knowledge_base 工具查询知识库中的行政规章制度。\n"
            "回答要求：\n"
            "1. 涵盖申请条件、所需材料、办理流程和注意事项\n"
            "2. 政策信息请以学校官方发布为准\n"
        ),
    },
]


def get_agent_prompt(agent_name: str) -> str | None:
    """获取指定智能体的系统提示词。"""
    for a in AGENT_DEFINITIONS:
        if a["name"] == agent_name:
            return a["system_prompt"]
    return None


def get_agent_descriptions() -> str:
    """获取所有子智能体的描述文本（用于 Supervisor 的系统提示词）。"""
    lines = ["可用子智能体："]
    for a in AGENT_DEFINITIONS:
        lines.append(f"- {a['name']}：{a['description']}")
    return "\n".join(lines)


async def _delegate_to_agent(agent_name: str, task: str) -> str:
    """
    将任务委托给指定的专业子智能体执行。
    子智能体拥有独立的 ReAct 循环，会自主调用工具并返回结果。
    """
    from app.agents.react import ReActAgent
    from app.llm.client import LLMClient
    from app.config import get_settings
    from app.tools import get_tool_registry

    prompt = get_agent_prompt(agent_name)
    if not prompt:
        return f"错误：不存在名为 '{agent_name}' 的智能体。可用：academic, life, admin"

    settings = get_settings()
    llm = LLMClient(settings)
    registry = get_tool_registry()

    # 子智能体只使用 KB 和校园工具，不包含委托工具（避免循环）
    sub_agent_tools = []
    for schema in registry.openai_schemas():
        func_name = schema.get("function", {}).get("name", "")
        if func_name != "delegate_to_agent":
            sub_agent_tools.append(schema)

    agent = ReActAgent(
        name=agent_name,
        system_prompt=prompt,
        llm_client=llm,
        tool_registry=registry,
        tools=sub_agent_tools,
    )

    result = await agent.run(
        user_message=task,
        max_steps=6,
    )

    final = result.answer
    if result.tool_calls > 0:
        final = f"[{agent_name}] {final}"
    return final


def _delegate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "enum": ["academic", "life", "admin"],
                "description": "目标子智能体名称：academic（学术）、life（生活）、admin（行政）",
            },
            "task": {
                "type": "string",
                "description": "需要委托执行的具体任务描述",
            },
        },
        "required": ["agent_name", "task"],
    }


def register_delegation_tool() -> None:
    """注册智能体委托工具到全局注册表。"""
    registry = get_tool_registry()
    registry.register(
        ToolDef(
            name="delegate_to_agent",
            description="将任务委托给专业子智能体执行。根据问题类型选择合适的智能体：academic-学术、life-生活、admin-行政",
            parameters=_delegate_schema(),
            function=_delegate_to_agent,
        )
    )
