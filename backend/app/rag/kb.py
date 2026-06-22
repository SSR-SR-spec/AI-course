"""
知识库核心模块：文档向量化、FAISS 索引读写、语义检索。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.chunking import chunk_text
from app.rag.document_loader import RawDocument
from app.utils import ensure_dir, stable_doc_id


@dataclass
class KBItem:
    """知识库中的一条文本块记录。"""
    doc_id: str
    source: str
    chunk_id: str
    text: str
    meta: dict[str, Any]


class CampusKB:
    """
    校园知识库：管理文档的向量化存储与语义检索。
    核心流程：
    - 使用 SentenceTransformer 将文本转为稠密向量
    - 使用 FAISS 进行余弦相似度搜索
    - 自动检测 embedding 模型变更并在索引维度不匹配时重建
    """

    def __init__(
        self,
        *,
        kb_dir: str,
        embedding_model: str,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        chunk_strategy: str = "recursive",
    ):
        self.kb_dir = kb_dir
        self.embedding_model_name = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunk_strategy = chunk_strategy
        self._embedder: SentenceTransformer | None = None
        self._use_local_hash: bool = False
        self._local_hash_dim: int = 1024

        self._index: faiss.Index | None = None
        self._items: list[KBItem] = []

        self._index_path = os.path.join(kb_dir, "faiss.index")
        self._items_path = os.path.join(kb_dir, "items.jsonl")
        self._meta_path = os.path.join(kb_dir, "meta.json")

    @property
    def embedder(self) -> SentenceTransformer:
        if self._use_local_hash:
            raise RuntimeError("本地哈希模式不暴露 SentenceTransformer 对象。")

        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                raise RuntimeError(
                    f"无法加载 embedding 模型 {self.embedding_model_name!r}。\n"
                    "常见原因：网络中断导致缓存不完整、镜像未生效、或与当前目录下同名文件夹冲突。\n"
                    "处理建议：\n"
                    "  1) 删除本机 HuggingFace 缓存里该模型目录后重试（通常在 "
                    "用户目录 .cache\\huggingface\\hub 下，文件夹名含 models--sentence-transformers--...）；\n"
                    "  2) 在 backend 目录执行：python scripts/download_embedding_model.py\n"
                    "     将 settings.yaml 中 rag.embedding_model 改为脚本打印的本地路径；\n"
                    "  3) 确认已安装：pip install sentence-transformers torch；\n"
                    "  4) 临时可改回 embedding_model: local_hash（离线哈希，无需下载）。\n"
                    f"原始错误：{type(e).__name__}: {e}"
                ) from e
        return self._embedder

    def load(self) -> None:
        """从磁盘加载 FAISS 索引和文本块数据到内存。"""
        ensure_dir(self.kb_dir)
        self._maybe_reset_if_embedding_changed()

        if os.path.exists(self._items_path):
            self._items = []
            with open(self._items_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self._items.append(
                        KBItem(
                            doc_id=obj["doc_id"],
                            source=obj["source"],
                            chunk_id=obj["chunk_id"],
                            text=obj["text"],
                            meta=obj.get("meta", {}),
                        )
                    )
        else:
            self._items = []

        if os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)
        else:
            self._index = None

    def _maybe_reset_if_embedding_changed(self) -> None:
        if not os.path.exists(self._meta_path):
            return
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return

        prev = meta.get("embedding_model")
        curr = self.embedding_model_name
        if prev and prev != curr:
            self._reset_storage(reason=f"embedding 模型已变更：{prev} -> {curr}")

    def _reset_storage(self, *, reason: str) -> None:
        """清空知识库的所有持久化文件（索引、文本块数据、元信息）。"""
        for p in (self._index_path, self._items_path, self._meta_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self._index = None
        self._items = []

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

    def stats(self) -> dict[str, Any]:
        if self._index is not None:
            dim = int(self._index.d)
        else:
            if self._use_local_hash or self.embedding_model_name == "local_hash":
                dim = self._local_hash_dim
            else:
                dim = int(self.embedder.get_sentence_embedding_dimension())
        return {
            "documents": len({it.doc_id for it in self._items}),
            "chunks": len(self._items),
            "embedding_dim": dim,
            "kb_dir": os.path.abspath(self.kb_dir),
        }

    def ingest(self, docs: list[RawDocument]) -> tuple[int, int]:
        ensure_dir(self.kb_dir)
        self.load()

        added_docs = 0
        added_chunks = 0

        new_items: list[KBItem] = []
        new_texts: list[str] = []

        seen_doc_ids = {it.doc_id for it in self._items}

        for doc in docs:
            text = (doc.text or "").strip()
            if not text:
                continue
            doc_id = stable_doc_id(doc.source, text)
            if doc_id in seen_doc_ids:
                continue

            added_docs += 1
            embedder_for_chunk = None
            if self._chunk_strategy == "semantic":
                embedder_for_chunk = self.embedder
            chunks = chunk_text(
                text,
                strategy=self._chunk_strategy,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                embedder=embedder_for_chunk,
            )
            for ch in chunks:
                item = KBItem(
                    doc_id=doc_id,
                    source=doc.source,
                    chunk_id=ch.chunk_id,
                    text=ch.text,
                    meta={**doc.meta, **ch.meta},
                )
                new_items.append(item)
                new_texts.append(ch.text)
            added_chunks += len(chunks)

        if not new_items:
            return 0, 0

        vecs = self._embed_texts(new_texts)
        dim = vecs.shape[1]
        self._ensure_index(dim)
        assert self._index is not None

        faiss.normalize_L2(vecs)
        self._index.add(vecs)

        self._items.extend(new_items)
        self._persist()
        self.load()
        return added_docs, added_chunks

    def retrieve(self, query: str, *, top_k: int = 5) -> list[tuple[KBItem, float]]:
        self.load()
        if self._index is None or len(self._items) == 0:
            return []
        qv = self._embed_texts([query])
        faiss.normalize_L2(qv)
        scores, idxs = self._index.search(qv, top_k)
        out: list[tuple[KBItem, float]] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(self._items):
                continue
            out.append((self._items[i], float(score)))
        return out

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.embedding_model_name == "local_hash":
            self._use_local_hash = True

        if self._use_local_hash:
            arr = _hash_embed(texts, dim=self._local_hash_dim)
        else:
            try:
                vecs = self.embedder.encode(texts, normalize_embeddings=False)
                arr = np.asarray(vecs, dtype="float32")
            except Exception:
                self._use_local_hash = True
                arr = _hash_embed(texts, dim=self._local_hash_dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _persist(self) -> None:
        assert self._index is not None
        faiss.write_index(self._index, self._index_path)

        with open(self._items_path, "w", encoding="utf-8") as f:
            for it in self._items:
                f.write(
                    json.dumps(
                        {
                            "doc_id": it.doc_id,
                            "source": it.source,
                            "chunk_id": it.chunk_id,
                            "text": it.text,
                            "meta": it.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"embedding_model": self.embedding_model_name},
                f,
                ensure_ascii=False,
                indent=2,
            )


_TOKEN_RE = re.compile(r"[一-鿿]+|[a-zA-Z0-9]+", re.UNICODE)


def _hash_embed(texts: list[str], *, dim: int) -> np.ndarray:
    """本地哈希嵌入：离线环境的后备方案，语义质量远低于 SentenceTransformer。"""
    mat = np.zeros((len(texts), dim), dtype="float32")
    for r, t in enumerate(texts):
        toks = _TOKEN_RE.findall(t.lower())
        if not toks:
            continue
        for tok in toks:
            h = _stable_hash(tok)
            mat[r, h % dim] += 1.0
    return mat


def _stable_hash(s: str) -> int:
    """FNV-1a 32 位哈希算法，将字符串稳定映射到整数。"""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)
