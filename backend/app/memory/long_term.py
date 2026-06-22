"""
长期记忆模块 — 从对话中提取关键信息并持久化存储，
支持跨会话的用户画像和知识积累。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils import ensure_dir


_INFO_PATTERNS: list[tuple[str, str, str]] = [
    # (正则表达式, 类别, 属性名)
    (r"(?:我是|我叫|我的名字)\s*(.+?)(?:大学|学生|的|，|。|$)", "user_profile", "name"),
    (r"(\d+)[年级级]", "user_profile", "grade"),
    (r"(计算机|会计|金融|法学|经济|管理|设计|英语|教育|医学|艺术|文学|新闻|数学|物理|化学|生物|工程|中文|历史|哲学|体育)\s*(?:专业|系|学院)", "user_profile", "major"),
    (r"学号[：:]?\s*(\d+)", "user_profile", "student_id"),
    (r"(\d{4})年入学", "user_profile", "enrollment_year"),
]


class LongTermMemory:
    """
    长期记忆：从对话中提取事实并持久化。
    每个用户（session_id）一个 JSONL 文件，存储结构化记忆条目。
    """

    def __init__(self, base_dir: str):
        self.base_dir = os.path.join(base_dir, "memory")
        ensure_dir(self.base_dir)

    def _path(self, session_id: str) -> str:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("_", "-"))
        if not safe:
            safe = "default"
        return os.path.join(self.base_dir, f"{safe}.jsonl")

    def add_memory(
        self,
        session_id: str,
        category: str,
        key: str,
        value: Any,
        source: str = "conversation",
    ) -> None:
        """写入一条长期记忆。"""
        p = self._path(session_id)
        entry = {
            "ts": datetime.now().isoformat(),
            "category": category,
            "key": key,
            "value": value,
            "source": source,
        }
        # 如果相同 key 的记忆已存在，覆盖
        existing = self.get_memories(session_id)
        filtered = [e for e in existing if not (e["category"] == category and e["key"] == key)]
        filtered.append(entry)
        with open(p, "w", encoding="utf-8") as f:
            for e in filtered:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def get_memories(self, session_id: str) -> list[dict[str, Any]]:
        """读取某用户的所有长期记忆。"""
        p = self._path(session_id)
        if not os.path.exists(p):
            return []
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def get_profile(self, session_id: str) -> dict[str, str]:
        """获取用户画像（方便拼接成 system prompt）。"""
        profile = {}
        for m in self.get_memories(session_id):
            if m["category"] == "user_profile":
                profile[m["key"]] = str(m["value"])
        return profile

    def format_profile_context(self, session_id: str) -> str:
        """格式化用户画像文本，用于注入 agent 的 system prompt。"""
        profile = self.get_profile(session_id)
        if not profile:
            return ""
        parts = []
        if "name" in profile:
            parts.append(f"用户姓名：{profile['name']}")
        if "grade" in profile:
            parts.append(f"年级：{profile['grade']}年级")
        if "major" in profile:
            parts.append(f"专业：{profile['major']}")
        if "student_id" in profile:
            parts.append(f"学号：{profile['student_id']}")
        if "enrollment_year" in profile:
            parts.append(f"入学年份：{profile['enrollment_year']}")
        return "已知用户信息：\n" + "\n".join(parts) if parts else ""

    def extract_and_store(self, session_id: str, user_message: str) -> dict[str, str]:
        """从用户消息中提取信息并存储。返回本次提取到的信息。"""
        extracted = {}
        for pattern_raw, category, key in _INFO_PATTERNS:
            m = re.search(pattern_raw, user_message)
            if m:
                value = m.group(1).strip()
                extracted[key] = value
                self.add_memory(session_id, category, key, value)
        return extracted
