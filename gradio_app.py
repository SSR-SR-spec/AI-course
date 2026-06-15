from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import httpx


BACKEND_URL = "http://127.0.0.1:8000"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "hbue-logo.webp"


def _sidebar_brand_html() -> str:
    if LOGO_PATH.is_file():
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_html = (
            f'<img src="data:image/webp;base64,{logo_b64}" '
            'class="sidebar-logo" alt="湖北经济学院" />'
        )
    else:
        logo_html = '<div class="sidebar-brand-icon">🎓</div>'
    return f"""
    <div class="sidebar-brand">
        {logo_html}
        <div class="sidebar-brand-text">Campus AI-HBUE</div>
    </div>
    """


def _pretty_json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _content_to_text(item)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    if isinstance(content, dict):
        for key in ("text", "content", "value", "label"):
            if key in content:
                text = _content_to_text(content.get(key))
                if text:
                    return text
        return ""
    return str(content).strip()


def _session_title_from_history(session_id: str, history: list[dict[str, str]]) -> str:
    for item in history:
        if item.get("role") == "user":
            text = _content_to_text(item.get("content"))
            if text:
                return text[:24] + ("..." if len(text) > 24 else "")
    return f"新对话"


def _session_choices(
    sessions: dict[str, list[dict[str, str]]],
    session_order: list[str],
) -> list[tuple[str, str]]:
    return [
        (_session_title_from_history(sid, sessions.get(sid, [])), sid)
        for sid in session_order
    ]


def call_chat(
    session_id: str,
    message: str,
    history: list[dict[str, str]] | None,
    sessions: dict[str, list[dict[str, str]]] | None,
    session_order: list[str] | None,
) -> tuple[list[dict[str, str]], str, str, dict[str, list[dict[str, str]]], list[str], gr.update]:
    history = history or []
    message = (message or "").strip()
    sessions = sessions or {}
    session_order = session_order or []
    sid = (session_id or "").strip() or "default"
    if sid not in session_order:
        session_order.append(sid)
    sessions[sid] = history

    if not message:
        return (
            history,
            "",
            "",
            sessions,
            session_order,
            gr.update(choices=_session_choices(sessions, session_order), value=sid),
        )

    payload = {"session_id": sid, "message": message, "metadata": {}}
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{BACKEND_URL}/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    answer = data.get("final_answer", "")
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    sessions[sid] = history

    debug = _pretty_json(
        {
            "citations": data.get("citations", []),
            "audit": data.get("audit"),
            "turn_id": data.get("turn_id"),
        }
    )
    return (
        history,
        debug,
        "",
        sessions,
        session_order,
        gr.update(choices=_session_choices(sessions, session_order), value=sid),
    )


def clear_chat(
    session_id: str,
    sessions: dict[str, list[dict[str, str]]] | None,
    session_order: list[str] | None,
) -> tuple[list[dict[str, str]], str, dict[str, list[dict[str, str]]], list[str], gr.update]:
    sessions = sessions or {}
    session_order = session_order or []
    sid = (session_id or "").strip() or "default"
    if sid not in session_order:
        session_order.append(sid)
    sessions[sid] = []
    return (
        [],
        "",
        sessions,
        session_order,
        gr.update(choices=_session_choices(sessions, session_order), value=sid),
    )


def _new_session(
    sessions: dict[str, list[dict[str, str]]] | None,
    session_order: list[str] | None,
) -> tuple[str, list[dict[str, str]], str, dict[str, list[dict[str, str]]], list[str], gr.update]:
    sessions = sessions or {}
    session_order = session_order or []
    sid = f"s-{uuid.uuid4().hex[:8]}"
    sessions[sid] = []
    session_order.append(sid)
    return (
        sid,
        [],
        "",
        sessions,
        session_order,
        gr.update(choices=_session_choices(sessions, session_order), value=sid),
    )


def switch_session(
    selected_sid: str,
    sessions: dict[str, list[dict[str, str]]] | None,
    session_order: list[str] | None,
) -> tuple[str, list[dict[str, str]], str, dict[str, list[dict[str, str]]], list[str], gr.update]:
    sessions = sessions or {}
    session_order = session_order or []
    sid = (selected_sid or "").strip() or "default"
    if sid not in session_order:
        session_order.append(sid)
    history = sessions.get(sid, [])
    sessions[sid] = history
    return (
        sid,
        history,
        "",
        sessions,
        session_order,
        gr.update(choices=_session_choices(sessions, session_order), value=sid),
    )


def toggle_sidebar(visible: bool) -> tuple[bool, gr.update, gr.update]:
    next_visible = not visible
    return (
        next_visible,
        gr.update(visible=next_visible),
        gr.update(value="«" if next_visible else "☰"),
    )


APP_CSS = """
/* ── Global reset ── */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #f4f4f5 !important;
}
footer { display: none !important; }

/* ── App shell ── */
.app-shell {
    min-height: 100vh;
    gap: 0 !important;
    margin: 0 !important;
}

/* ── Sidebar ── */
.sidebar {
    background: #171717 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 10px 10px !important;
    gap: 8px !important;
    border-right: 1px solid #2a2a2a;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 6px 8px;
    border-bottom: 1px solid #2e2e2e;
    margin-bottom: 0;
    flex-shrink: 0;
}

.sidebar-logo {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: contain;
    flex-shrink: 0;
    background: #ffffff;
    padding: 2px;
    box-sizing: border-box;
}

.sidebar-brand-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: linear-gradient(135deg, #10b981, #059669);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.sidebar-brand-text {
    color: #f4f4f5;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.sidebar-toolbar {
    gap: 8px !important;
    align-items: stretch !important;
    flex-shrink: 0 !important;
    margin: 0 !important;
}

.sidebar-btn button,
.new-chat-btn button,
.clear-btn button {
    width: 100% !important;
    min-height: 40px !important;
    height: 40px !important;
    max-height: 40px !important;
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #e4e4e7 !important;
    border-radius: 8px !important;
    padding: 0 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    transition: background 0.15s, border-color 0.15s !important;
    justify-content: center !important;
    align-items: center !important;
    box-shadow: none !important;
}
.sidebar-btn button:hover,
.new-chat-btn button:hover,
.clear-btn button:hover {
    background: #3f3f46 !important;
    border-color: #52525b !important;
    color: #fafafa !important;
}

.session-section-label {
    color: #71717a;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 6px 2px;
    flex-shrink: 0;
}

.session-list-wrap {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Session list (Radio) */
.session-list fieldset {
    border: none !important;
    padding: 0 !important;
    gap: 2px !important;
}
.session-list .wrap {
    flex-direction: column !important;
    gap: 2px !important;
}
.session-list label {
    width: 100% !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    min-height: 40px !important;
    color: #d4d4d8 !important;
    font-size: 13px !important;
    border: none !important;
    background: transparent !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    max-width: 100% !important;
}
.session-list label:hover {
    background: #27272a !important;
}
.session-list label.selected,
.session-list input:checked + label {
    background: #3f3f46 !important;
    color: #fafafa !important;
}
.session-list .wrap > label > span {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

.sidebar-toggle-btn button {
    background: transparent !important;
    border: 1px solid #e4e4e7 !important;
    color: #52525b !important;
    border-radius: 8px !important;
    min-width: 40px !important;
    width: 40px !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    padding: 0 !important;
    font-size: 16px !important;
    box-shadow: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.sidebar-toggle-btn button:hover {
    background: #f4f4f5 !important;
}

/* ── Main panel ── */
.main-panel {
    background: #f4f4f5 !important;
    min-height: 100vh;
    padding: 0 !important;
    gap: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

.main-header {
    align-items: center !important;
    gap: 12px !important;
    padding: 8px 16px !important;
    background: #f4f4f5;
    border-bottom: 1px solid #e4e4e7;
    flex-shrink: 0;
    min-height: 56px !important;
    margin: 0 !important;
}

.main-header-title {
    font-size: 16px;
    font-weight: 600;
    color: #18181b;
}

.main-header-sub {
    font-size: 12px;
    color: #71717a;
    margin-top: 2px;
}

.chat-area {
    flex: 1 1 auto !important;
    padding: 0 !important;
    overflow: hidden !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

#chatbot {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    flex: 1 1 auto !important;
    height: calc(100vh - 210px) !important;
    min-height: 360px !important;
    max-height: calc(100vh - 210px) !important;
}
#chatbot > div,
#chatbot .wrapper,
#chatbot .component-wrap {
    height: 100% !important;
    min-height: inherit !important;
}
#chatbot .wrapper {
    background: transparent !important;
}
#chatbot .message-row {
    padding: 10px 28px !important;
    max-width: 1080px;
    margin: 0 auto;
    font-size: 15.5px !important;
    line-height: 1.65 !important;
}
#chatbot .message.user {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #18181b !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
#chatbot .message.bot {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #18181b !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* ── Input dock ── */
.input-dock {
    padding: 8px 16px 10px !important;
    background: #f4f4f5;
    border-top: 1px solid #e4e4e7;
    flex-shrink: 0;
}

.input-box {
    max-width: 1080px;
    margin: 0 auto;
    background: #ffffff !important;
    border: 1px solid #d4d4d8 !important;
    border-radius: 12px !important;
    padding: 4px 4px 4px 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
    align-items: stretch !important;
    gap: 8px !important;
    min-height: 48px !important;
}
.input-box:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.15), 0 2px 12px rgba(0,0,0,0.08) !important;
}
.input-box textarea {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 15px !important;
    padding: 10px 0 !important;
    resize: none !important;
    min-height: 40px !important;
    max-height: 120px !important;
}
.input-box .wrap {
    border: none !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    min-height: 40px !important;
}
.send-btn {
    align-self: stretch !important;
    display: flex !important;
}
.send-btn button {
    background: #10b981 !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0 18px !important;
    min-height: 40px !important;
    height: 100% !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.send-btn button:hover {
    background: #059669 !important;
}

.input-hint {
    text-align: center;
    font-size: 11px;
    color: #a1a1aa;
    margin-top: 6px;
    max-width: 1080px;
    margin-left: auto;
    margin-right: auto;
}

/* ── Debug panel ── */
.debug-panel {
    max-width: 1080px;
    margin: 0 auto 12px !important;
    padding: 0 24px !important;
}
.debug-panel .label-wrap { display: none !important; }
.debug-panel textarea {
    font-family: monospace !important;
    font-size: 12px !important;
    background: #fafafa !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 10px !important;
    color: #52525b !important;
}
"""

APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.zinc,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f4f4f5",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_label_text_weight="600",
    button_primary_background_fill="#10b981",
    button_primary_background_fill_hover="#059669",
)


with gr.Blocks(title="Campus Assistant", fill_height=True) as demo:
    sessions_state = gr.State({"s1": []})
    session_order_state = gr.State(["s1"])
    sidebar_visible_state = gr.State(True)

    session_id = gr.Textbox(value="s1", visible=False)

    with gr.Row(elem_classes=["app-shell"], equal_height=True):
        # ── Sidebar ──
        with gr.Column(elem_classes=["sidebar"], scale=0, min_width=240, visible=True) as sidebar_col:
            gr.HTML(_sidebar_brand_html())
            with gr.Row(elem_classes=["sidebar-toolbar"]):
                new_session = gr.Button("+ 新对话", elem_classes=["sidebar-btn", "new-chat-btn"], scale=1, size="sm")
                clear = gr.Button("清空", elem_classes=["sidebar-btn", "clear-btn"], scale=1, size="sm")
            gr.HTML('<div class="session-section-label">历史对话</div>')
            with gr.Column(elem_classes=["session-list-wrap"]):
                session_list = gr.Radio(
                    choices=[("新对话", "s1")],
                    value="s1",
                    label=None,
                    show_label=False,
                    elem_classes=["session-list"],
                )

        # ── Main ──
        with gr.Column(elem_classes=["main-panel"], scale=1):
            with gr.Row(elem_classes=["main-header"]):
                sidebar_toggle = gr.Button("«", elem_classes=["sidebar-toggle-btn"], scale=0, min_width=40, size="sm")
                gr.HTML(
                    """
                    <div>
                        <div class="main-header-title">校园智能助手</div>
                        <div class="main-header-sub">基于 RAG 知识库 · 多智能体协作</div>
                    </div>
                    """
                )

            with gr.Column(elem_classes=["chat-area"]):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=480,
                    min_height=360,
                    show_label=False,
                    container=False,
                    layout="bubble",
                    buttons=["copy"],
                    placeholder="有什么可以帮你的？询问课程、生活、行政等各类校园问题",
                )

            with gr.Column(elem_classes=["input-dock"]):
                with gr.Row(elem_classes=["input-box"]):
                    msg = gr.Textbox(
                        label=None,
                        placeholder="输入你的问题，Enter 发送，Shift+Enter 换行...",
                        scale=9,
                        container=False,
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        autofocus=True,
                    )
                    send = gr.Button("发送 ↑", variant="primary", scale=0, min_width=88, size="sm", elem_classes=["send-btn"])

                gr.HTML('<div class="input-hint">AI 回答仅供参考，重要事项请以官方通知为准</div>')

                with gr.Accordion("引用与审核详情", open=False, elem_classes=["debug-panel"]):
                    debug_out = gr.Textbox(label=None, lines=8, show_label=False)

    sidebar_toggle.click(
        toggle_sidebar,
        inputs=[sidebar_visible_state],
        outputs=[sidebar_visible_state, sidebar_col, sidebar_toggle],
    )

    send.click(
        call_chat,
        inputs=[session_id, msg, chatbot, sessions_state, session_order_state],
        outputs=[chatbot, debug_out, msg, sessions_state, session_order_state, session_list],
    )
    msg.submit(
        call_chat,
        inputs=[session_id, msg, chatbot, sessions_state, session_order_state],
        outputs=[chatbot, debug_out, msg, sessions_state, session_order_state, session_list],
    )
    clear.click(
        clear_chat,
        inputs=[session_id, sessions_state, session_order_state],
        outputs=[chatbot, debug_out, sessions_state, session_order_state, session_list],
    )
    new_session.click(
        _new_session,
        inputs=[sessions_state, session_order_state],
        outputs=[session_id, chatbot, debug_out, sessions_state, session_order_state, session_list],
    )
    session_list.change(
        switch_session,
        inputs=[session_list, sessions_state, session_order_state],
        outputs=[session_id, chatbot, debug_out, sessions_state, session_order_state, session_list],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=APP_THEME,
        css=APP_CSS,
    )
