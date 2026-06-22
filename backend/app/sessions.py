"""
会话存储模块 — 将聊天历史持久化到本地 JSONL 文件。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from app.utils import ensure_dir, now_ms


Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class StoredMessage:
    """持久化存储的对话消息。"""
    role: Role
    content: str
    ts_ms: int


class FileSessionStore:
    """
    基于文件的会话存储。
    每个 session_id 对应一个 JSONL 文件，存储在 base_dir/sessions/ 目录下。
    适用于演示和小规模场景，生产环境建议替换为 Redis / SQLite。
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.sessions_dir = os.path.join(base_dir, "sessions")
        ensure_dir(self.sessions_dir)

    def _path(self, session_id: str) -> str:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("_", "-", "."))
        if not safe:
            safe = "default"
        return os.path.join(self.sessions_dir, f"{safe}.jsonl")

    def append(self, session_id: str, role: Role, content: str) -> None:
        """向指定会话追加一条消息。"""
        p = self._path(session_id)
        ensure_dir(os.path.dirname(p))
        msg = {"role": role, "content": content, "ts_ms": now_ms()}
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def load_tail(self, session_id: str, *, max_messages: int) -> list[StoredMessage]:
        """
        加载指定会话最近的若干条消息。
        简单实现：读取整个文件后取尾部（适用于演示规模）。
        """
        p = self._path(session_id)
        if not os.path.exists(p):
            return []
        msgs: list[StoredMessage] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msgs.append(
                        StoredMessage(
                            role=obj["role"],
                            content=obj["content"],
                            ts_ms=int(obj.get("ts_ms", 0) or 0),
                        )
                    )
                except Exception:
                    continue
        return msgs[-max_messages:]

    def clear(self, session_id: str) -> None:
        p = self._path(session_id)
        if os.path.exists(p):
            os.remove(p)
