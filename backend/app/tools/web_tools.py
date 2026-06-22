"""
联网搜索工具 — Agent 可通过它搜索互联网获取实时信息。
使用 Bing 搜索（国内可访问，免费，无需 API Key）。
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="This package.*duckduckgo_search.*renamed")

import asyncio
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.tools import ToolDef, get_tool_registry

BING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


async def _web_search(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新信息。"""
    if max_results < 1:
        max_results = 1
    if max_results > 10:
        max_results = 10

    # 先试 DuckDuckGo
    result = await _try_duckduckgo(query, max_results)
    if result:
        return result

    # 回退 Bing
    result = await _try_bing(query, max_results)
    if result:
        return result

    return (
        f"联网搜索不可用。当前网络环境可能无法访问搜索引擎。\n"
        f"搜索关键词：{query}\n"
        f"建议检查网络连接，或使用更具体的关键词。"
    )


async def _try_duckduckgo(query: str, max_results: int) -> str | None:
    """尝试 DuckDuckGo 搜索。"""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"[{i+1}] {title}\n    来源：{href}\n    {body[:200]}")
        if results:
            return "【联网搜索结果】\n\n" + "\n\n".join(results)
    except Exception:
        pass
    return None


async def _try_bing(query: str, max_results: int) -> str | None:
    """回退到 Bing 搜索（国内可用）。"""
    def _sync_search():
        try:
            url = "https://www.bing.com/search"
            params = {"q": query, "count": max_results, "setlang": "zh-Hans"}
            resp = requests.get(url, params=params, headers=BING_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for i, li in enumerate(soup.select("#b_results > li.b_algo"), 1):
                if i > max_results:
                    break
                title_el = li.select_one("h2 a")
                snippet_el = li.select_one(".b_caption p")
                title = title_el.get_text(strip=True) if title_el else ""
                href = title_el.get("href", "") if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                if title:
                    results.append(f"[{i}] {title}\n    来源：{href}\n    {snippet[:200]}")
            if results:
                return "【联网搜索结果】\n\n" + "\n\n".join(results)
            return None
        except Exception:
            return None

    return await asyncio.to_thread(_sync_search)


def _web_search_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，像百度/Google 一样自然输入",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量（1-10），默认 5",
                "default": 5,
            },
        },
        "required": ["query"],
    }


def register_web_tools() -> None:
    """注册联网搜索工具到全局注册表。"""
    registry = get_tool_registry()
    registry.register(
        ToolDef(
            name="web_search",
            description="搜索互联网获取实时信息。适用于查询最新新闻、知识库未覆盖的内容、校外信息、实时资讯等。知识库已有的信息不要用它，请用 search_knowledge_base",
            parameters=_web_search_schema(),
            function=_web_search,
        )
    )
