"""
工具系统 — 注册、定义和调用可被 LLM 调用的工具。
提供 OpenAI-compatible function calling 格式的 schema 生成。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    """LLM 发起的工具调用请求。"""
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """工具执行结果。"""
    tool_call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class ToolDef:
    """工具定义：名称、描述、参数 JSON Schema 以及对应的执行函数。"""
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Coroutine[Any, Any, str]]

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 格式的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """全局工具注册表，管理所有可用工具并提供查找和执行能力。"""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        if tool.name in self._tools:
            return  # 已注册则跳过，确保幂等
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        """返回所有工具的 OpenAI function calling schema 列表。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """执行指定工具并返回结果。"""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                output="",
                success=False,
                error=f"未知工具：{name}，可用工具：{list(self._tools.keys())}",
            )
        try:
            output = await tool.function(**arguments)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                output=str(output),
                success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                output="",
                success=False,
                error=f"工具 {name} 执行失败：{type(e).__name__}: {e}",
            )

    async def execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResult]:
        """批量执行工具调用。"""
        results = []
        for tc in tool_calls:
            result = await self.execute(tc.id, tc.name, tc.arguments)
            results.append(result)
        return results


# 全局工具注册表单例
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表（懒加载）。"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
