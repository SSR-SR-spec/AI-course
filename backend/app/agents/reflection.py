"""
反思与纠错引擎 — 由 LLM 驱动的回答质量审核与改进。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReviewResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: float = 1.0


REVIEW_PROMPT = """\
你是一个校园智能助手的回答质量审核员。请严格审核以下对话，从这些维度评价：

1. 完整性：回答是否覆盖了用户问题的所有方面
2. 准确性：回答是否基于提供的工具结果，没有编造信息
3. 引用性：涉及政策/制度问题时，是否引用了知识库内容
4. 清晰度：回答是否简洁易懂

以严格的 JSON 格式输出（不要包含其他内容）：
{"passed": true/false, "score": 0.0-1.0, "issues": ["问题1"], "suggestions": ["建议1"]}

pass=false 当且仅当存在明显的问题（编造信息、严重遗漏、违反规定）。
score 反映整体质量，0.6 以上为可接受。
"""


IMPROVE_PROMPT = """\
你是一个校园智能助手。用户对你的回答不满意，审核反馈如下。
请根据反馈改进回答，确保：
1. 直接回答用户问题
2. 引用知识库信息
3. 简洁清晰

只输出改进后的回答，不要包含其他内容。
"""


class ReflectionEngine:
    """
    反思引擎 — 由 LLM 驱动，无硬编码规则。
    review() 让 LLM 审核回答质量，improve() 让 LLM 改进回答。
    """

    def __init__(self, llm_client, use_mock: bool = False):
        self.llm_client = llm_client
        self.use_mock = use_mock
        self.max_revisions = 2

    async def review(
        self,
        query: str,
        answer: str,
        tool_results: list[str] | None = None,
    ) -> ReviewResult:
        """LLM 驱动的回答审核。"""
        if self.use_mock:
            return ReviewResult(passed=True, score=1.0)

        from app.config import get_settings
        settings = get_settings()

        tool_summary = ""
        if tool_results:
            tool_summary = "\n工具调用结果：\n" + "\n".join(tool_results[-3:])

        user_msg = (
            f"用户问题：{query}\n"
            f"回答：{answer}{tool_summary}"
        )

        try:
            res = await self.llm_client.complete(
                system=REVIEW_PROMPT,
                user=user_msg,
            )
            import json as j
            data = j.loads(res.content)
            return ReviewResult(
                passed=data.get("passed", True),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                score=float(data.get("score", 0.5)),
            )
        except Exception:
            return ReviewResult(passed=True, score=0.7)

    async def improve(self, query: str, answer: str, review: ReviewResult) -> str:
        """LLM 驱动的回答改进。"""
        if self.use_mock or review.passed:
            return answer

        user_msg = (
            f"用户问题：{query}\n"
            f"原有回答：{answer}\n"
            f"审核问题：{', '.join(review.issues)}\n"
            f"改进建议：{', '.join(review.suggestions)}"
        )

        try:
            res = await self.llm_client.complete(
                system=IMPROVE_PROMPT,
                user=user_msg,
            )
            return res.content
        except Exception:
            return answer
