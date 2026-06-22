"""
Pydantic 数据模型定义 — 请求/响应的结构校验和文档说明。
包含原有的 RAG 模型和新增的 Agent 模型。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ===== 请求模型 =====

class ChatMessage(BaseModel):
    """单条对话消息。"""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """聊天请求体。"""
    session_id: str | None = None
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestFolderRequest(BaseModel):
    """文档导入请求体。"""
    folder_path: str


# ===== Agent 请求模型 =====

class AgentChatRequest(BaseModel):
    """智能体聊天请求体。"""
    session_id: str | None = None
    message: str
    use_reflection: bool = True
    use_planning: bool = True


# ===== 响应模型 =====

class Citation(BaseModel):
    """检索到的引用片段，记录来源、位置和相似度。"""
    doc_id: str
    source: str
    chunk_id: str
    score: float
    snippet: str
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    """单个智能体的回答结果。"""
    agent: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AuditResult(BaseModel):
    """回答质量审核结果。"""
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


class ChatResponse(BaseModel):
    """聊天响应（结构化返回）。"""
    turn_id: str
    session_id: str
    intent: str
    routed_agents: list[str]
    final_answer: str
    citations: list[Citation]
    audit: AuditResult
    debug: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """文档导入结果响应。"""
    added_documents: int
    added_chunks: int
    kb_dir: str
    files_scanned: int = 0
    note: str | None = None


class KBStats(BaseModel):
    """知识库统计信息。"""
    documents: int
    chunks: int
    embedding_dim: int
    kb_dir: str


# ===== Agent 响应模型 =====

class AgentStepInfo(BaseModel):
    """Agent 执行步骤信息。"""
    role: str = Field(description="thought / tool_call / tool_result / final")
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """Agent 执行追踪信息。"""
    steps: list[AgentStepInfo] = Field(default_factory=list)
    tool_calls_count: int = 0
    reflection_used: bool = False
    planning_used: bool = False


class AgentChatResponse(BaseModel):
    """智能体聊天响应。"""
    turn_id: str
    session_id: str
    answer: str
    trace: AgentTrace | None = None
    profile_updated: dict[str, str] = Field(default_factory=dict, description="本次提取到的用户信息")
