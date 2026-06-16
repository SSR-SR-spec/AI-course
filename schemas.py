"""
Pydantic 数据模型定义 — 请求/响应的结构校验
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ===== 请求模型 =====

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str | None = None     # 会话 ID，不传则用默认
    message: str                       # 用户消息
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestFolderRequest(BaseModel):
    """文档导入请求"""
    folder_path: str                   # 要扫描的本地文件夹路径


# ===== 响应模型 =====

class Citation(BaseModel):
    """检索到的引用片段"""
    doc_id: str
    source: str                        # 来源文档路径
    chunk_id: str                      # 文本块编号
    score: float                       # 向量相似度分数
    snippet: str                       # 原文片段
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    """单个智能体的回答"""
    agent: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AuditResult(BaseModel):
    """回答质量审核结果"""
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


class ChatResponse(BaseModel):
    """聊天响应（结构化返回）"""
    turn_id: str                       # 本轮唯一 ID
    session_id: str
    intent: str                        # 识别到的意图
    routed_agents: list[str]           # 参与回答的智能体列表
    final_answer: str                  # 最终回答
    citations: list[Citation]          # 引用的知识片段
    audit: AuditResult                 # 质量审核结果
    debug: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """文档导入结果"""
    added_documents: int
    added_chunks: int
    kb_dir: str
    files_scanned: int = 0
    note: str | None = None            # 提示信息（如无新增文档的原因）


class KBStats(BaseModel):
    """知识库统计信息"""
    documents: int
    chunks: int
    embedding_dim: int
    kb_dir: str

