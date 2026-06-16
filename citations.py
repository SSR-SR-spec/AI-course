from __future__ import annotations

import os
import re

from app.schemas import Citation

_SENTENCE_END = re.compile(r"[。！？!?；;]$")


def source_display_name(source: str) -> str:
    """Return a human-readable document name from a file path or URL."""
    source = (source or "").strip()
    if not source:
        return "未知文档"
    name = os.path.basename(source.replace("\\", "/"))
    return name or source


def is_complete_sentence(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 6:
        return False
    return bool(_SENTENCE_END.search(text))


def extract_complete_passage(snippet: str, *, max_lines: int = 6) -> str | None:
    """
    Extract a complete sentence or paragraph from a retrieved chunk.
    Drops leading/trailing fragments caused by chunk boundaries.
    """
    text = (snippet or "").strip()
    if len(text) < 8:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        selected: list[str] = []
        for line in lines:
            if not selected and not is_complete_sentence(line) and len(lines) > 1:
                continue
            if selected and not is_complete_sentence(line):
                break
            selected.append(line)
            if len(selected) >= max_lines:
                break
        if selected:
            return "\n".join(selected)

    flat = " ".join(text.split())
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;])", flat) if p.strip()]
    if not parts:
        return flat if len(flat) >= 12 else None

    selected_parts: list[str] = []
    for part in parts:
        if not selected_parts and not is_complete_sentence(part) and len(parts) > 1:
            continue
        if selected_parts and not is_complete_sentence(part):
            break
        selected_parts.append(part)
        if len(selected_parts) >= max_lines:
            break

    if selected_parts:
        return "".join(selected_parts)
    return flat if len(flat) >= 12 and is_complete_sentence(flat) else None


def citations_with_passages(citations: list[Citation]) -> list[tuple[Citation, str]]:
    """Keep only citations that contain a complete quotable passage."""
    out: list[tuple[Citation, str]] = []
    seen: set[tuple[str, str]] = set()
    for c in citations:
        passage = extract_complete_passage(c.snippet)
        if not passage:
            continue
        key = (source_display_name(c.source), passage)
        if key in seen:
            continue
        seen.add(key)
        out.append((c, passage))
    return out


def format_citations_footer(citations: list[Citation]) -> str:
    """Build a markdown block listing documents and complete quoted passages."""
    pairs = citations_with_passages(citations)
    if not pairs:
        return ""

    lines = ["---", "**参考资料**", ""]
    for idx, (c, passage) in enumerate(pairs, start=1):
        doc_name = source_display_name(c.source)
        lines.append(f"**[{idx}] {doc_name}**")
        for line in passage.splitlines():
            lines.append(f"> {line}")
        lines.append("")

    return "\n".join(lines).rstrip()


def append_citations_to_answer(answer: str, citations: list[Citation]) -> str:
    """Append formatted citations only when relevant passages were retrieved."""
    body = (answer or "").rstrip()
    body = re.sub(r"\n*引用[：:]\s*(\[\d+\]\s*)+$", "", body, flags=re.MULTILINE)
    body = re.sub(r"\n*引用[：:]\s*\[[\d\[\],\s]+\]\s*$", "", body)
    body = re.sub(r"\n*---\s*\n\*\*参考资料\*\*[\s\S]*$", "", body)

    if not citations_with_passages(citations):
        return body

    footer = format_citations_footer(citations)
    if not footer:
        return body
    if not body:
        return footer
    return f"{body}\n\n{footer}"
