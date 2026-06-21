"""
知识库工具 — 封装知识库检索功能为 LLM 可调用的工具。
"""

from __future__ import annotations

import json
from typing import Any

from app.tools import ToolDef, get_tool_registry


def _search_kb_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": '校园知识库搜索关键词，如"选课要求""奖学金申请条件"',
            },
            "top_k": {
                "type": "integer",
                "description": "返回的匹配片段数量（1-10）",
                "default": 5,
            },
        },
        "required": ["query"],
    }


def _get_kb_stats_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "detail": {
                "type": "string",
                "description": "查询的详细主题（可选）",
                "default": "",
            }
        },
    }


async def _search_kb(query: str, top_k: int = 5) -> str:
    """在校园知识库中搜索与 query 相关的内容片段。"""
    # 延迟导入以避免循环依赖
    from app.main import kb

    if kb is None:
        return "知识库未加载，无法检索。请先确认知识库路径正确并重启服务。"
    try:
        if query:
            hits = kb.retrieve(query, top_k=top_k)
        else:
            return "搜索查询为空。"
        if not hits:
            return f"未找到与「{query}」相关的内容。"
        lines = [f"找到 {len(hits)} 条相关结果："]
        for i, (item, score) in enumerate(hits, 1):
            snippet = (item.text or "")[:200].replace("\n", " ")
            source = item.source.split("\\")[-1].split("/")[-1]
            lines.append(f"\n[{i}] 来源：{source} | 相关度：{score:.4f}")
            lines.append(f"    {snippet}...")
        return "\n".join(lines)
    except Exception as e:
        return f"知识库检索异常：{type(e).__name__}: {e}"


async def _get_kb_stats(detail: str = "") -> str:
    """查询知识库的统计信息。"""
    from app.main import kb

    if kb is None:
        return "知识库未加载。"
    try:
        stats = kb.stats()
        parts = [
            f"知识库统计：",
            f"  - 文档数：{stats.get('documents', 'N/A')}",
            f"  - 文本块数：{stats.get('chunks', 'N/A')}",
            f"  - 向量维度：{stats.get('embedding_dim', 'N/A')}",
            f"  - 存储路径：{stats.get('kb_dir', 'N/A')}",
        ]
        if detail:
            parts.append(f"\n查询主题：{detail}")
        return "\n".join(parts)
    except Exception as e:
        return f"获取知识库统计失败：{type(e).__name__}: {e}"


def register_kb_tools() -> None:
    """注册所有知识库相关工具到全局注册表。"""
    registry = get_tool_registry()
    registry.register(
        ToolDef(
            name="search_knowledge_base",
            description="在校园知识库中搜索相关政策、规章制度、办事流程等信息。",
            parameters=_search_kb_schema(),
            function=_search_kb,
        )
    )
    registry.register(
        ToolDef(
            name="get_knowledge_base_stats",
            description="获取知识库的统计信息，如文档数量、文本块数量等。",
            parameters=_get_kb_stats_schema(),
            function=_get_kb_stats,
        )
    )
