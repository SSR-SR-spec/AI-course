"""
配置管理：从 YAML 文件 + 环境变量读取设置
优先级：环境变量 > settings.local.yaml > settings.yaml > 内置默认值
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    """应用设置模型，包含 LLM、RAG 等所有配置项"""
    app_name: str = "campus-llm-mas-rag"

    # ===== LLM 配置 =====
    llm_provider: str = "mock"  # "mock" | "openai_compatible"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"
    llm_timeout_s: int = 120
    llm_max_output_tokens: int = 800

    # Storage / cache
    redis_url: str | None = None

    # RAG
    kb_dir: str = "./kb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_max_chars_per_snippet: int = 700
    rag_max_context_chars: int = 6000
    rag_max_snippets: int = 3
    rag_min_score: float = 0.35

    # Chunking (used during ingest)
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 150

    # Hugging Face Hub 镜像（国内用 hf-mirror.com）
    huggingface_hub_endpoint: str | None = None


def get_settings() -> Settings:
    """
    Settings precedence:
    1) Environment variables (optional override)
    2) `app/settings.yaml` + `app/settings.local.yaml` (recommended for local dev)
    3) Built-in defaults
    """
    file_cfg = _load_yaml_settings_files()
    merged = _flatten_settings_dict(file_cfg)

    # Built-in defaults (used when a key is missing from file/env)
    defaults: dict[str, Any] = {
        "app_name": "campus-llm-mas-rag",
        "llm_provider": "mock",
        "openai_api_key": None,
        "openai_base_url": None,
        "openai_model": "gpt-4o",
        "llm_timeout_s": 120,
        "llm_max_output_tokens": 800,
        "redis_url": None,
        "kb_dir": "./kb",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "rag_top_k": 5,
        "rag_max_chars_per_snippet": 700,
        "rag_max_context_chars": 6000,
        "rag_max_snippets": 3,
        "rag_min_score": 0.35,
        "rag_chunk_size": 900,
        "rag_chunk_overlap": 150,
        "huggingface_hub_endpoint": None,
    }

    # Merge file into defaults (file wins over defaults)
    data = {**defaults, **merged}

    # Env overrides (env wins over file)
    data["app_name"] = _getenv("APP_NAME", data["app_name"]) or data["app_name"]
    data["llm_provider"] = _getenv("LLM_PROVIDER", data["llm_provider"]) or data["llm_provider"]
    data["openai_api_key"] = _getenv("OPENAI_API_KEY", data["openai_api_key"])
    data["openai_base_url"] = _getenv("OPENAI_BASE_URL", data["openai_base_url"])
    data["openai_model"] = _getenv("OPENAI_MODEL", data["openai_model"]) or data["openai_model"]
    data["llm_timeout_s"] = int(_getenv("LLM_TIMEOUT_S", str(data["llm_timeout_s"])))
    data["llm_max_output_tokens"] = int(
        _getenv("LLM_MAX_OUTPUT_TOKENS", str(data["llm_max_output_tokens"]))
    )
    data["redis_url"] = _getenv("REDIS_URL", data["redis_url"])
    data["kb_dir"] = _getenv("KB_DIR", data["kb_dir"]) or data["kb_dir"]
    data["embedding_model"] = _getenv("EMBEDDING_MODEL", data["embedding_model"]) or data[
        "embedding_model"
    ]
    data["rag_top_k"] = int(_getenv("RAG_TOP_K", str(data["rag_top_k"])))
    data["rag_max_chars_per_snippet"] = int(
        _getenv("RAG_MAX_SNIPPET_CHARS", str(data["rag_max_chars_per_snippet"]))
    )
    data["rag_max_context_chars"] = int(
        _getenv("RAG_MAX_CONTEXT_CHARS", str(data["rag_max_context_chars"]))
    )
    data["rag_max_snippets"] = int(_getenv("RAG_MAX_SNIPPETS", str(data["rag_max_snippets"])))
    data["rag_min_score"] = float(_getenv("RAG_MIN_SCORE", str(data["rag_min_score"])))
    data["rag_chunk_size"] = int(_getenv("RAG_CHUNK_SIZE", str(data["rag_chunk_size"])))
    data["rag_chunk_overlap"] = int(_getenv("RAG_CHUNK_OVERLAP", str(data["rag_chunk_overlap"])))
    data["huggingface_hub_endpoint"] = _getenv(
        "HF_ENDPOINT", data.get("huggingface_hub_endpoint")
    )

    # Safety: if user configured openai_compatible but forgot the key, fall back to mock.
    if data["llm_provider"] == "openai_compatible" and not (data.get("openai_api_key") or ""):
        data["llm_provider"] = "mock"

    _apply_huggingface_hub_endpoint(data.get("huggingface_hub_endpoint"))

    return Settings.model_validate(data)


def _apply_huggingface_hub_endpoint(url: str | None) -> None:
    """Set HF hub endpoint for model downloads (mirror-friendly, e.g. China networks)."""
    import os

    if not url:
        return
    u = str(url).strip().rstrip("/")
    if u:
        os.environ["HF_ENDPOINT"] = u


def apply_hf_hub_endpoint_before_imports() -> None:
    """
    Must run before `app.rag.kb` is imported: that module loads `sentence_transformers`,
    which loads `huggingface_hub`. If HF_ENDPOINT is set too late, requests still go to
    huggingface.co and mirrors in settings.yaml are ignored.
    """
    import os

    try:
        cfg = _load_yaml_settings_files()
        flat = _flatten_settings_dict(cfg)
        url = flat.get("huggingface_hub_endpoint") or os.getenv("HF_ENDPOINT")
        _apply_huggingface_hub_endpoint(url if isinstance(url, str) else None)
    except Exception:
        # Non-fatal: get_settings() will try again later
        pass


def _getenv(key: str, default: str | None) -> str | None:
    import os

    v = os.getenv(key)
    return v if v is not None and v != "" else default


def _app_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_yaml_settings_files() -> dict[str, Any]:
    base = _app_dir() / "settings.yaml"
    local = _app_dir() / "settings.local.yaml"

    merged: dict[str, Any] = {}
    for p in (base, local):
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid settings file (expected mapping): {p}")
        merged = _deep_merge(merged, obj)
    return merged


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _flatten_settings_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Map nested YAML to flat Settings fields.
    """
    llm = cfg.get("llm") or {}
    rag = cfg.get("rag") or {}
    redis = cfg.get("redis") or {}

    out: dict[str, Any] = {}
    if isinstance(cfg.get("app_name"), str):
        out["app_name"] = cfg["app_name"]

    if isinstance(llm, dict):
        if "provider" in llm:
            out["llm_provider"] = llm["provider"]
        if "openai_api_key" in llm:
            out["openai_api_key"] = llm["openai_api_key"] or None
        if "openai_base_url" in llm:
            out["openai_base_url"] = llm["openai_base_url"] or None
        if "openai_model" in llm:
            out["openai_model"] = llm["openai_model"]
        if "timeout_s" in llm:
            out["llm_timeout_s"] = int(llm["timeout_s"])
        if "max_output_tokens" in llm:
            out["llm_max_output_tokens"] = int(llm["max_output_tokens"])

    if isinstance(redis, dict) and "url" in redis:
        out["redis_url"] = redis["url"] or None

    if isinstance(rag, dict):
        if "kb_dir" in rag:
            out["kb_dir"] = rag["kb_dir"]
        if "embedding_model" in rag:
            out["embedding_model"] = rag["embedding_model"]
        if "top_k" in rag:
            out["rag_top_k"] = int(rag["top_k"])
        if "max_chars_per_snippet" in rag:
            out["rag_max_chars_per_snippet"] = int(rag["max_chars_per_snippet"])
        if "max_context_chars" in rag:
            out["rag_max_context_chars"] = int(rag["max_context_chars"])
        if "max_snippets" in rag:
            out["rag_max_snippets"] = int(rag["max_snippets"])
        if "min_score" in rag:
            out["rag_min_score"] = float(rag["min_score"])
        if "chunk_size" in rag:
            out["rag_chunk_size"] = int(rag["chunk_size"])
        if "chunk_overlap" in rag:
            out["rag_chunk_overlap"] = int(rag["chunk_overlap"])
        if "huggingface_hub_endpoint" in rag:
            v = rag.get("huggingface_hub_endpoint")
            out["huggingface_hub_endpoint"] = (str(v).strip() if v else None) or None

    return out

