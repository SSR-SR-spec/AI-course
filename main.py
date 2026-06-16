from __future__ import annotations

# Apply HF mirror (settings.yaml) before sentence_transformers/huggingface_hub load via kb.
from app.citations import append_citations_to_answer, citations_with_passages
from app.config import apply_hf_hub_endpoint_before_imports

apply_hf_hub_endpoint_before_imports()

import logging
import os
from pathlib import Path
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.llm.client import LLMClient
from app.rag.document_loader import iter_documents_in_folder
from app.rag.kb import CampusKB, KBItem
from app.sessions import FileSessionStore
from app.schemas import (
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
kb = CampusKB(
    kb_dir=settings.kb_dir,
    embedding_model=settings.embedding_model,
    chunk_size=settings.rag_chunk_size,
    chunk_overlap=settings.rag_chunk_overlap,
)
kb.load()
llm = LLMClient(settings)
session_store = FileSessionStore(base_dir=settings.kb_dir)

# Very small in-memory trace store for demo.
# If you need persistence, swap with Redis / SQLite.
TRACE: dict[str, dict[str, Any]] = {}

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ingest_note(
    *,
    files_scanned: int,
    added_docs: int,
    added_chunks: int,
    kb_chunks_before: int,
) -> str | None:
    if files_scanned == 0:
        return (
            "目录下未发现可导入文件。请确认路径正确，且包含后缀为 "
            ".txt / .md / .html / .htm / .pdf 的文件。"
        )
    if added_docs == 0 and added_chunks == 0:
        if kb_chunks_before == 0:
            return (
                f"已扫描 {files_scanned} 个文件，但未新增：正文为空，或无法提取文本"
                "（例如扫描版 PDF 无文字层）。请检查文件内容是否为可复制文本。"
            )
        return (
            f"已扫描 {files_scanned} 个文件，但未新增：与知识库中已有内容重复（按文档内容去重），"
            "或本次文件正文为空。若需强制重建，请更换 rag.kb_dir 或删除该 kb 目录后重试。"
        )
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ts": now_ms()}


@app.post("/ingest/local-folder", response_model=IngestResponse)
def ingest_local_folder(req: IngestFolderRequest) -> IngestResponse:
    folder = req.folder_path
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Not a folder: {folder}")

    try:
        before = kb.stats()
        docs = list(iter_documents_in_folder(folder))
        added_docs, added_chunks = kb.ingest(docs)
        kb_dir_abs = os.path.abspath(settings.kb_dir)
        note = _ingest_note(
            files_scanned=len(docs),
            added_docs=added_docs,
            added_chunks=added_chunks,
            kb_chunks_before=int(before.get("chunks", 0) or 0),
        )
        return IngestResponse(
            added_documents=added_docs,
            added_chunks=added_chunks,
            kb_dir=kb_dir_abs,
            files_scanned=len(docs),
            note=note,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {type(e).__name__}: {e}",
        )


@app.get("/kb/stats", response_model=KBStats)
def kb_stats() -> KBStats:
    s = kb.stats()
    return KBStats(**s)


def _route_intent(message: str) -> tuple[str, list[str]]:
    m = message.lower()
    if any(k in m for k in ["选课", "培养", "学分", "绩点", "课程", "毕业", "教务"]):
        return "academic", ["academic_agent"]
    if any(k in m for k in ["食堂", "菜单", "图书馆", "座位", "地图", "校车", "宿舍"]):
        return "life", ["life_agent"]
    if any(k in m for k in ["奖学金", "请假", "报销", "审批", "行政", "资助", "流程"]):
        return "admin", ["admin_agent"]
    return "general", ["academic_agent", "life_agent", "admin_agent"]


def _select_rag_citations(
    hits: list[tuple[KBItem, float]],
) -> tuple[list[Citation], list[str]]:
    citations: list[Citation] = []
    ctx_lines: list[str] = []
    used_chars = 0
    max_snippets = max(1, int(settings.rag_max_snippets))
    max_ctx = max(500, int(settings.rag_max_context_chars))
    min_score = float(settings.rag_min_score)

    rank = 0
    for it, score in hits:
        if score < min_score:
            continue
        full_text = (it.text or "").strip()
        if not full_text:
            continue

        rank += 1
        ctx_snippet = full_text[: settings.rag_max_chars_per_snippet]
        line = f"[{rank}] source={it.source} chunk={it.chunk_id} score={score:.4f}\n{ctx_snippet}"
        if used_chars + len(line) + 2 > max_ctx:
            break
        citations.append(
            Citation(
                doc_id=it.doc_id,
                source=it.source,
                chunk_id=it.chunk_id,
                score=score,
                snippet=full_text,
                meta=it.meta,
            )
        )
        ctx_lines.append(line)
        used_chars += len(line) + 2
        if len(citations) >= max_snippets:
            break

    return citations, ctx_lines


def _chat_system_prompt(*, has_citations: bool) -> str:
    base = (
        "你是校园知识服务平台中的专业智能体。"
        "严禁复述/粘贴整篇文档；只输出与问题直接相关的要点，并尽量控制在 10 行以内。"
        "不要输出引用片段之外的“全文”。如果用户要求全文，请拒绝并建议用户直接打开原文文件。"
        "输出要简洁；不要在回答末尾自行列出引用编号或引用详情，系统会自动附上文档出处。"
    )
    if has_citations:
        return (
            base
            + "必须优先依据给定的【引用片段】回答；"
            "当引用片段不足以支持结论时，明确说明不确定并建议咨询官方渠道。"
        )
    return (
        base
        + "当前未检索到与问题相关的知识库文档，请明确告知用户知识库中暂无相关资料，"
        "不要编造答案，也不要声称存在参考文献。"
    )


async def _answer_with_rag(*, agent_name: str, message: str) -> tuple[str, list[Citation], dict[str, Any]]:
    hits = kb.retrieve(message, top_k=settings.rag_top_k)
    citations, ctx_lines = _select_rag_citations(hits)

    system = _chat_system_prompt(has_citations=bool(citations))
    user = message

    # Attach short conversation context (stored server-side).
    # We keep this outside the citations block to avoid encouraging verbatim paste.
    history = session_store.load_tail(agent_name + "__" + message[:0], max_messages=0)  # noop placeholder
    # Actual history will be passed in from chat() to avoid double file reads.

    res = await llm.complete(system=system, user=user)
    relevant = [c for c, _ in citations_with_passages(citations)]
    debug = {"provider": res.raw.get("provider"), "retrieved": len(hits), "relevant": len(relevant)}
    return append_citations_to_answer(res.content, relevant), relevant, debug


def _build_user_prompt(*, agent_name: str, message: str, ctx_lines: list[str], history_lines: list[str]) -> str:
    parts: list[str] = []
    parts.append(f"你的角色：{agent_name}")
    if history_lines:
        parts.append("【对话上下文（最近几轮）】")
        parts.extend(history_lines)
    parts.append(f"用户问题：{message}")
    parts.append("【引用片段】")
    parts.append("\n\n".join(ctx_lines) if ctx_lines else "（无）")
    return "\n".join(parts)


def _audit(*, message: str, answer: str, citations: list[Citation]) -> AuditResult:
    issues: list[str] = []
    if not citations:
        issues.append("未检索到可引用的知识片段，答案可能缺乏依据。")
    if len(answer.strip()) < 10:
        issues.append("答案过短，可能未覆盖用户问题。")
    passed = len(issues) == 0
    suggested_fix = None
    if not passed:
        suggested_fix = (
            "建议：先调用 /ingest/local-folder 导入校园制度/指南文档；"
            "再提问以生成带引用的回答。"
        )
    return AuditResult(passed=passed, issues=issues, suggested_fix=suggested_fix)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or "default"
    turn_id = new_turn_id()

    try:
        intent, agents = _route_intent(req.message)

        # For demo: pick the first routed agent to answer, then audit it.
        agent_name = agents[0]
        # Load recent history for this session (server-side persisted).
        past = session_store.load_tail(session_id, max_messages=12)
        history_lines: list[str] = []
        for m in past:
            if m.role == "user":
                history_lines.append(f"用户：{m.content}")
            elif m.role == "assistant":
                history_lines.append(f"助手：{m.content}")

        hits = kb.retrieve(req.message, top_k=settings.rag_top_k)
        citations, ctx_lines = _select_rag_citations(hits)
        system = _chat_system_prompt(has_citations=bool(citations))
        user_prompt = _build_user_prompt(
            agent_name=agent_name,
            message=req.message,
            ctx_lines=ctx_lines,
            history_lines=history_lines[-12:],  # limit prompt size
        )

        res = await llm.complete(system=system, user=user_prompt)
        answer = res.content
        relevant_citations = [c for c, _ in citations_with_passages(citations)]
        debug = {
            "provider": res.raw.get("provider"),
            "retrieved": len(hits),
            "relevant": len(relevant_citations),
            "min_score": settings.rag_min_score,
        }
        audit = _audit(message=req.message, answer=answer, citations=relevant_citations)

        final = append_citations_to_answer(answer, relevant_citations)
        resp = ChatResponse(
            turn_id=turn_id,
            session_id=session_id,
            intent=intent,
            routed_agents=agents,
            final_answer=final,
            citations=relevant_citations,
            audit=audit,
            debug=debug,
        )

        # Persist conversation
        session_store.append(session_id, "user", req.message)
        session_store.append(session_id, "assistant", final)

        TRACE[turn_id] = {
            "request": req.model_dump(),
            "response": resp.model_dump(),
        }
        return resp
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("chat failed: %s\n%s", e, tb)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {type(e).__name__}: {e}",
        )


@app.get("/trace/{turn_id}")
def trace(turn_id: str) -> dict[str, Any]:
    obj = TRACE.get(turn_id)
    if not obj:
        raise HTTPException(status_code=404, detail="turn_id not found")
    return obj


# ========== 前端静态文件托管 ==========
# 将 frontend/ 目录挂载到根路径，使聊天页面可直接通过浏览器访问
# 重要：必须放在所有 API 路由之后，避免覆盖 API 路径
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

