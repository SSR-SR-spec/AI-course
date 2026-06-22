"""
工具函数：时间戳、ID 生成、文档去重哈希、目录创建。
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid


def now_ms() -> int:
    """返回当前时间的毫秒级时间戳。"""
    return int(time.time() * 1000)


def new_turn_id() -> str:
    """生成一次对话轮次的唯一标识 ID（UUID hex 格式）。"""
    return uuid.uuid4().hex


def stable_doc_id(source: str, content: str) -> str:
    """
    基于文档来源路径和内容生成稳定的哈希 ID。
    用于文档去重：相同来源且相同内容的文档不会重复导入知识库。
    """
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\n")
    h.update(content.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def ensure_dir(path: str) -> None:
    """确保指定目录存在，若不存在则递归创建。"""
    os.makedirs(path, exist_ok=True)
