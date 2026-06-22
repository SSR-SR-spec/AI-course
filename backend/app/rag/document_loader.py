"""
文档加载器 — 从本地文件夹读取支持格式的文档（纯文本、Markdown、HTML、PDF）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(frozen=True)
class RawDocument:
    """原始文档结构，包含来源路径、文本内容和元信息。"""
    source: str
    text: str
    meta: dict


SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".pdf"}


def iter_documents_in_folder(folder_path: str) -> Iterable[RawDocument]:
    """
    递归遍历文件夹，读取并转换所有支持的文档文件。
    支持格式：.txt / .md / .html / .htm / .pdf
    """
    for root, _, files in os.walk(folder_path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            full_path = os.path.join(root, name)
            yield load_document(full_path)


def load_document(path: str) -> RawDocument:
    """根据文件扩展名调用相应的读取函数，返回 RawDocument 对象。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        text = _read_text(path)
    elif ext in (".html", ".htm"):
        text = _read_html(path)
    elif ext == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")

    source = os.path.abspath(path)
    meta = {"path": source, "ext": ext}
    return RawDocument(source=source, text=text, meta=meta)


def _read_text(path: str) -> str:
    """读取纯文本文件（.txt / .md）。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_html(path: str) -> str:
    """读取 HTML 文件，提取可见文本，去除 script/style 标签内容。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return _normalize(text)


def _read_pdf(path: str) -> str:
    """读取 PDF 文件，逐页提取文本。"""
    reader = PdfReader(path)
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return _normalize("\n".join(parts))


def _normalize(text: str) -> str:
    """规范化文本：去除每行首尾空格，移除空行。"""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)
