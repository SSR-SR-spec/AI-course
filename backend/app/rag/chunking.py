"""
文本分块模块 — 提供多种分块策略：
1. SimpleChunker  — 基于字符数滑动窗口（原策略，保留向后兼容）
2. RecursiveChunker — 递归分块：段落 → 句子 → 字符
3. SemanticChunker — 基于嵌入向量的语义边界检测（最新策略）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Chunk:
    """文本块结构，包含编号、文本内容和元信息。"""
    chunk_id: str
    text: str
    meta: dict


# =============================================================================
# 策略 1：简单字符滑动窗口（原策略）
# =============================================================================

def chunk_text_simple(
    text: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """基于字符数的简单分块（带重叠）。"""
    clean = text.strip()
    if not clean:
        return []
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end]
        chunk_id = f"c{idx:06d}"
        chunks.append(Chunk(chunk_id=chunk_id, text=piece, meta={"start": start, "end": end}))
        idx += 1
        if end == len(clean):
            break
        start = max(0, end - chunk_overlap)
    return chunks


# =============================================================================
# 策略 2：递归分块（段落 → 句子 → 字符）
# =============================================================================

_SENTENCE_SPLITTER = re.compile(r"(?<=[。！？!?；;])\s*")
_PARAGRAPH_SPLITTER = re.compile(r"\n\s*\n")


def chunk_text_recursive(
    text: str,
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    递归分块：优先按段落分割，如果段落仍太长则按句子分割。
    最大限度地保持语义完整性。
    """
    clean = text.strip()
    if not clean:
        return []

    paragraphs = _split_paragraphs(clean)

    chunks: list[Chunk] = []
    buffer = ""
    idx = 0

    for para in paragraphs:
        # 如果段落长度超过 chunk_size，按句子拆分
        if len(para) > chunk_size:
            # 先把缓冲区的内容写入
            if buffer:
                chunks.append(Chunk(
                    chunk_id=f"c{idx:06d}", text=buffer.strip(),
                    meta={"method": "para_combine"},
                ))
                idx += 1
                buffer = ""
            # 将长段落按句子拆分成多个块
            sentence_chunks = _split_long_text(para, chunk_size)
            for sc in sentence_chunks:
                chunks.append(Chunk(
                    chunk_id=f"c{idx:06d}", text=sc,
                    meta={"method": "sentence_split"},
                ))
                idx += 1
        else:
            # 段落未超长，尝试合并到缓冲区
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(Chunk(
                        chunk_id=f"c{idx:06d}", text=buffer,
                        meta={"method": "para_combine"},
                    ))
                    idx += 1
                buffer = para

    # 处理剩余缓冲区
    if buffer:
        chunks.append(Chunk(
            chunk_id=f"c{idx:06d}", text=buffer,
            meta={"method": "para_combine"},
        ))

    # 如果没有分块（纯文本无段落标记），退化为简单分块
    if not chunks:
        return chunk_text_simple(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """按空行分割段落。"""
    raw = _PARAGRAPH_SPLITTER.split(text)
    return [p.strip() for p in raw if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """按句号/问号/感叹号分割句子。"""
    raw = _SENTENCE_SPLITTER.split(text)
    return [s.strip() for s in raw if s.strip()]


def _split_long_text(text: str, max_size: int) -> list[str]:
    """将长文本按句子拆分为不超过 max_size 的块。"""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        # 无法按句子拆分，退化为字符级
        return _char_split(text, max_size)

    chunks: list[str] = []
    buffer = ""
    for sent in sentences:
        if len(sent) > max_size:
            # 单句超长，强制字符拆分
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_char_split(sent, max_size))
        else:
            candidate = (buffer + sent).strip() if buffer else sent
            if len(candidate) <= max_size:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = sent

    if buffer:
        chunks.append(buffer)

    return chunks


def _char_split(text: str, max_size: int) -> list[str]:
    """纯字符级拆分（兜底）。"""
    return [text[i:i + max_size] for i in range(0, len(text), max_size)]


# =============================================================================
# 策略 3：语义分块（基于嵌入向量的语义边界检测）
# =============================================================================

_MIN_CHUNK_CHARS = 50


def chunk_text_semantic(
    text: str,
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    embedder: Any = None,  # SentenceTransformer 实例
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """
    语义分块：将文本按语义边界切分，保证每个块内主题一致。
    流程：
    1. 先按段落和句子切分为候选片段
    2. 用嵌入模型编码相邻候选片段的语义相似度
    3. 在相似度低于阈值的位置切分

    Args:
        embedder: SentenceTransformer 实例（可选）。不提供时退化为递归分块。
        similarity_threshold: 语义边界阈值（0-1），越低越容易切分
    """
    clean = text.strip()
    if not clean:
        return []

    # 没有 embedder 时退化为递归分块
    if embedder is None:
        return chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 1. 切分为候选句群
    sentences = _split_sentences(clean)
    if len(sentences) <= 2:
        return chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2. 将小句子合并为候选块（每个至少 ~chunk_size/3 字符）
    candidate_groups = _merge_small_sentences(sentences, min_size=chunk_size // 3)

    # 3. 编码候选块的嵌入向量，检测语义边界
    try:
        batch_texts = [g for g in candidate_groups if g.strip()]
        if len(batch_texts) <= 1:
            return chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        embeddings = embedder.encode(batch_texts, normalize_embeddings=True)

        # 4. 计算相邻块的余弦相似度，在低点切分
        import numpy as np
        boundaries = [0]
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            if sim < similarity_threshold:
                boundaries.append(i + 1)

        boundaries.append(len(candidate_groups))

        # 5. 根据边界组装最终块
        idx = 0
        chunks: list[Chunk] = []
        for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
            merged = "".join(candidate_groups[b_start:b_end]).strip()
            if len(merged) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                chunk_id=f"c{idx:06d}",
                text=merged,
                meta={"method": "semantic", "segments": (b_start, b_end)},
            ))
            idx += 1

        # 如果语义分块没有产出（阈值太高），退化为递归分块
        if len(chunks) <= 1 and len(clean) > chunk_size:
            return chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        return chunks if chunks else chunk_text_recursive(text, chunk_size=chunk_size)

    except Exception:
        # 编码失败时安全退化为递归分块
        return chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _merge_small_sentences(sentences: list[str], min_size: int) -> list[str]:
    """将过短的句子合并到前一个句子中。"""
    groups: list[str] = []
    buf = ""
    for sent in sentences:
        if not sent:
            continue
        if not buf:
            buf = sent
        elif len(buf) < min_size:
            buf += sent
        else:
            groups.append(buf)
            buf = sent
    if buf:
        groups.append(buf)
    return groups


# =============================================================================
# 统一入口
# =============================================================================

CHUNK_STRATEGIES = {
    "simple": chunk_text_simple,
    "recursive": chunk_text_recursive,
    "semantic": chunk_text_semantic,
}


def chunk_text(
    text: str,
    *,
    strategy: str = "recursive",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    embedder: Any = None,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """
    统一分块入口，根据 strategy 参数选择分块策略。

    Args:
        strategy: "simple" | "recursive" | "semantic"
        embedder: 语义分块需要的 SentenceTransformer 实例
        similarity_threshold: 语义分块的边界敏感度（0-1）
    """
    if strategy == "simple":
        return chunk_text_simple(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "semantic":
        return chunk_text_semantic(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedder=embedder,
            similarity_threshold=similarity_threshold,
        )
    # 默认 recursive
    return chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
