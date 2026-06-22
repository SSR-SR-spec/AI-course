"""
校园智能助手 — FastAPI 入口
提供 RAG 问答和 Agent 智能体两套系统。
"""

from __future__ import annotations

from app.citations import append_citations_to_answer, citations_with_passages
from app.config import apply_hf_hub_endpoint_before_imports

apply_hf_hub_endpoint_before_imports()

import logging
import os
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.llm.client import LLMClient
from app.rag.document_loader import iter_documents_in_folder
from app.rag.kb import CampusKB
from app.sessions import FileSessionStore
from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentStepInfo,
    AgentTrace,
    AuditResult,
    ChatRequest,
    ChatResponse,
    Citation,
    IngestFolderRequest,
    IngestResponse,
    KBStats,
)
from app.utils import new_turn_id, now_ms

logger = logging.getLogger("campus")
settings = get_settings()

# ===== 初始化核心组件 =====
kb = CampusKB(
    kb_dir=settings.kb_dir,
    embedding_model=settings.embedding_model,
    chunk_size=settings.rag_chunk_size,
    chunk_overlap=settings.rag_chunk_overlap,
    chunk_strategy=settings.rag_chunk_strategy,
)
kb.load()
llm = LLMClient(settings)
session_store = FileSessionStore(base_dir=settings.kb_dir)

# ===== Agent 系统（延迟初始化） =====
_supervisor = None
_long_term_memory = None
_reflection_engine = None
_agent_initialized = False


def _init_agent_system():
    """初始化 Agent 系统：注册所有工具、创建 Supervisor 和记忆系统。"""
    global _supervisor, _long_term_memory, _reflection_engine, _agent_initialized
    if _agent_initialized:
        return

    from app.tools import get_tool_registry
    from app.tools.kb_tools import register_kb_tools
    from app.tools.campus_tools import register_campus_tools
    from app.tools.delegation import register_delegation_tool
    from app.tools.web_tools import register_web_tools
    register_kb_tools()
    register_campus_tools()
    register_delegation_tool()
    register_web_tools()

    from app.agents.supervisor import Supervisor
    _supervisor = Supervisor(llm_client=llm, use_mock=settings.llm_provider == "mock")

    from app.agents.reflection import ReflectionEngine
    _reflection_engine = ReflectionEngine(llm_client=llm, use_mock=settings.llm_provider == "mock")

    from app.memory.long_term import LongTermMemory
    _long_term_memory = LongTermMemory(base_dir=settings.kb_dir)

    _agent_initialized = True


app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRACE: dict[str, dict[str, Any]] = {}


# =============================================================================
# 健康检查
# =============================================================================


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ts": now_ms()}


# =============================================================================
# 知识库管理
# =============================================================================


@app.post("/ingest/local-folder", response_model=IngestResponse)
def ingest_local_folder(req: IngestFolderRequest) -> IngestResponse:
    """导入本地文件夹中的文档到知识库。"""
    folder = req.folder_path
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Not a folder: {folder}")
    try:
        before = kb.stats()
        docs = list(iter_documents_in_folder(folder))
        added_docs, added_chunks = kb.ingest(docs)
        kb_dir_abs = os.path.abspath(settings.kb_dir)

        note = None
        if len(docs) == 0:
            note = "目录下未发现可导入文件。请确认路径正确，且包含 .txt/.md/.html/.htm/.pdf 文件。"
        elif added_docs == 0 and added_chunks == 0 and int(before.get("chunks", 0) or 0) == 0:
            note = "文件内容为空或无法提取文本。"

        return IngestResponse(
            added_documents=added_docs,
            added_chunks=added_chunks,
            kb_dir=kb_dir_abs,
            files_scanned=len(docs),
            note=note,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {type(e).__name__}: {e}")


@app.get("/kb/stats", response_model=KBStats)
def kb_stats() -> KBStats:
    return KBStats(**kb.stats())


# =============================================================================
# 传统 RAG 聊天（向后兼容）
# =============================================================================


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """传统 RAG 聊天接口（保留向后兼容）。"""
    session_id = req.session_id or "default"
    turn_id = new_turn_id()
    try:
        intent = "general"
        agents = ["general"]

        history = session_store.load_tail(session_id, max_messages=12)
        history_lines = [
            f"{m.role}：{m.content}" for m in history
        ]

        hits = kb.retrieve(req.message, top_k=settings.rag_top_k)
        citations, ctx_lines = _select_rag_citations(hits)
        system = _chat_system_prompt(has_citations=bool(citations))
        user_prompt = _build_user_prompt(
            message=req.message,
            ctx_lines=ctx_lines,
            history_lines=history_lines[-12:],
        )

        res = await llm.complete(system=system, user=user_prompt)
        relevant_citations = [c for c, _ in citations_with_passages(citations)]
        final = append_citations_to_answer(res.content, relevant_citations)

        session_store.append(session_id, "user", req.message)
        session_store.append(session_id, "assistant", final)

        resp = ChatResponse(
            turn_id=turn_id,
            session_id=session_id,
            intent=intent,
            routed_agents=agents,
            final_answer=final,
            citations=relevant_citations,
            audit=AuditResult(passed=True),
            debug={"provider": res.raw.get("provider"), "retrieved": len(hits)},
        )
        TRACE[turn_id] = {"request": req.model_dump(), "response": resp.model_dump()}
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat failed: {type(e).__name__}: {e}")


# =============================================================================
# Agent 智能体聊天（真正的多智能体系统）
# =============================================================================


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    """
    智能体聊天 — 真正的 LLM 驱动的 Agent 系统。

    流程：
    1. 加载长期记忆（用户画像）
    2. Supervisor（ReActAgent）自主决策 → 调用工具 → 委托子智能体 → 合成回答
    3. ReflectionEngine 审核回答质量，必要时改进
    4. 提取本次对话信息到长期记忆
    5. 保存到会话历史
    """
    session_id = req.session_id or "default"
    turn_id = new_turn_id()

    try:
        _init_agent_system()
        assert _supervisor is not None
        assert _long_term_memory is not None
        assert _reflection_engine is not None

        # 1. 长期记忆
        profile_update = _long_term_memory.extract_and_store(session_id, req.message)
        user_profile = _long_term_memory.format_profile_context(session_id)

        # 2. 会话历史上下文
        past = session_store.load_tail(session_id, max_messages=6)
        context_lines = [
            f"{'用户' if m.role == 'user' else '助手'}：{m.content[:200]}"
            for m in past
        ]
        context_str = "\n".join(context_lines[-6:])

        # 3. Supervisor ReAct 循环（LLM 自主决策一切）
        result = await _supervisor.run(
            query=req.message,
            context=context_str,
            user_profile=user_profile,
            max_steps=10 if req.use_planning else 6,
        )

        answer = result.answer

        # 4. 反思审核（LLM 驱动的质量检查）
        if req.use_reflection:
            review = await _reflection_engine.review(
                query=req.message,
                answer=answer,
            )
            if not review.passed:
                logger.info("Reflection: answer needs improvement (%s)", "; ".join(review.issues))
                improved = await _reflection_engine.improve(
                    query=req.message,
                    answer=answer,
                    review=review,
                )
                if improved and len(improved) > len(answer) * 0.5:
                    answer = improved

        # 5. 持久化会话
        session_store.append(session_id, "user", req.message)
        session_store.append(session_id, "assistant", answer)

        # 6. 构建追踪
        trace_steps = []
        for step in result.steps:
            trace_steps.append(AgentStepInfo(
                role=step.role,
                content=step.content[:300],
                metadata=step.metadata,
            ))

        trace = AgentTrace(
            steps=trace_steps,
            tool_calls_count=result.tool_calls,
            reflection_used=req.use_reflection,
            planning_used=req.use_planning,
        )

        resp = AgentChatResponse(
            turn_id=turn_id,
            session_id=session_id,
            answer=answer,
            trace=trace,
            profile_updated=profile_update,
        )

        TRACE[turn_id] = {"request": req.model_dump(), "response": resp.model_dump()}
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.error("agent_chat failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Agent chat failed: {type(e).__name__}: {e}",
        )


@app.get("/trace/{turn_id}")
def trace(turn_id: str) -> dict[str, Any]:
    obj = TRACE.get(turn_id)
    if not obj:
        raise HTTPException(status_code=404, detail="turn_id not found")
    return obj


# =============================================================================
# 静态文件托管
# =============================================================================

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.exists():

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = _frontend_dir / full_path
        if not file_path.exists() or not file_path.is_file():
            file_path = _frontend_dir / "chat.html"
        if file_path.exists():
            return FileResponse(str(file_path))
        return JSONResponse(status_code=404, content={"detail": "Not Found"})


# =============================================================================
# RAG 辅助函数（保留给 /chat 端点）
# =============================================================================


def _select_rag_citations(
    hits: list[tuple[Any, float]],
) -> tuple[list[Citation], list[str]]:
    citations: list[Citation] = []
    ctx_lines: list[str] = []
    used_chars = 0
    max_snippets = max(1, int(settings.rag_max_snippets))
    max_ctx = max(500, int(settings.rag_max_context_chars))
    min_score = float(settings.rag_min_score)
    rank = 0
    for item, score in hits:
        if score < min_score:
            continue
        full_text = (item.text or "").strip()
        if not full_text:
            continue
        rank += 1
        ctx_snippet = full_text[: settings.rag_max_chars_per_snippet]
        line = f"[{rank}] source={item.source} score={score:.4f}\n{ctx_snippet}"
        if used_chars + len(line) + 2 > max_ctx:
            break
        citations.append(Citation(
            doc_id=item.doc_id, source=item.source, chunk_id=item.chunk_id,
            score=score, snippet=full_text, meta=item.meta,
        ))
        ctx_lines.append(line)
        used_chars += len(line) + 2
        if len(citations) >= max_snippets:
            break
    return citations, ctx_lines


def _chat_system_prompt(*, has_citations: bool) -> str:
    base = (
        "你是校园知识服务平台中的专业智能体。"
        "严禁复述整篇文档；只输出与问题直接相关的要点。"
        "不要输出引用片段以外的内容。输出要简洁；"
        "不要在回答末尾自行列出引用编号，系统会自动附上文档出处。"
    )
    if has_citations:
        return base + "必须优先依据给定的引用片段回答；当引用不足时说明不确定。"
    return base + "当前未检索到相关资料，请告知用户并不要编造答案。"


def _build_user_prompt(
    *,
    message: str,
    ctx_lines: list[str],
    history_lines: list[str],
) -> str:
    parts = []
    if history_lines:
        parts.append("【对话上下文】")
        parts.extend(history_lines)
    parts.append(f"用户问题：{message}")
    parts.append("【引用片段】")
    parts.append("\n\n".join(ctx_lines) if ctx_lines else "（无）")
    return "\n".join(parts)
