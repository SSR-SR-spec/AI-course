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
    llm_provider: str = "mock"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"
    llm_timeout_s: int = 120
    llm_max_output_tokens: int = 800

    # ===== 存储 / 缓存 =====
    redis_url: str | None = None

    # ===== RAG 配置 =====
    kb_dir: str = "./kb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_max_chars_per_snippet: int = 700
    rag_max_context_chars: int = 6000
    rag_max_snippets: int = 3
    rag_min_score: float = 0.35

    # ===== 文本分块配置 =====
    rag_chunk_strategy: str = "recursive"  # simple | recursive | semantic
    rag_chunk_size: int = 700
    rag_chunk_overlap: int = 100
    rag_similarity_threshold: float = 0.75  # 语义分块阈值（0-1）

    # ===== HuggingFace Hub 镜像 =====
    huggingface_hub_endpoint: str | None = None


def get_settings() -> Settings:
    """
    加载配置，优先级（高到低）：
    1) 环境变量（可选覆盖）
    2) app/settings.yaml + app/settings.local.yaml（推荐用于本地开发）
    3) 内置默认值
    """
    file_cfg = _load_yaml_settings_files()
    merged = _flatten_settings_dict(file_cfg)

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
        "rag_chunk_strategy": "recursive",
        "rag_chunk_size": 700,
        "rag_chunk_overlap": 100,
        "rag_similarity_threshold": 0.75,
        "huggingface_hub_endpoint": None,
    }

    data = {**defaults, **merged}

    # 环境变量覆盖（优先级最高）
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
    data["rag_chunk_strategy"] = _getenv("RAG_CHUNK_STRATEGY", data["rag_chunk_strategy"]) or data["rag_chunk_strategy"]
    data["rag_chunk_size"] = int(_getenv("RAG_CHUNK_SIZE", str(data["rag_chunk_size"])))
    data["rag_chunk_overlap"] = int(_getenv("RAG_CHUNK_OVERLAP", str(data["rag_chunk_overlap"])))
    data["rag_similarity_threshold"] = float(_getenv("RAG_SIMILARITY_THRESHOLD", str(data["rag_similarity_threshold"])))
    data["huggingface_hub_endpoint"] = _getenv(
        "HF_ENDPOINT", data.get("huggingface_hub_endpoint")
    )

    # 安全性检查：如果配置了 openai_compatible 但未提供密钥，降级为 mock
    if data["llm_provider"] == "openai_compatible" and not (data.get("openai_api_key") or ""):
        data["llm_provider"] = "mock"

    _apply_huggingface_hub_endpoint(data.get("huggingface_hub_endpoint"))

    return Settings.model_validate(data)


def _apply_huggingface_hub_endpoint(url: str | None) -> None:
    """设置 HuggingFace Hub 镜像地址（国内网络使用镜像站加速模型下载）。"""
    import os

    if not url:
        return
    u = str(url).strip().rstrip("/")
    if u:
        os.environ["HF_ENDPOINT"] = u


def apply_hf_hub_endpoint_before_imports() -> None:
    """
    在 sentence_transformers / huggingface_hub 导入之前调用。
    这些库在导入时会初始化 HTTP 客户端，若 HF_ENDPOINT 设晚了，
    请求仍会发往 huggingface.co 导致镜像配置失效。
    """
    import os

    try:
        cfg = _load_yaml_settings_files()
        flat = _flatten_settings_dict(cfg)
        url = flat.get("huggingface_hub_endpoint") or os.getenv("HF_ENDPOINT")
        _apply_huggingface_hub_endpoint(url if isinstance(url, str) else None)
    except Exception:
        pass


def _getenv(key: str, default: str | None) -> str | None:
    """读取环境变量，若未设置或为空则返回默认值。"""
    import os

    v = os.getenv(key)
    return v if v is not None and v != "" else default


def _app_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_yaml_settings_files() -> dict[str, Any]:
    """加载 settings.yaml 和 settings.local.yaml，合并为一个字典（local 覆盖 base）。"""
    base = _app_dir() / "settings.yaml"
    local = _app_dir() / "settings.local.yaml"

    merged: dict[str, Any] = {}
    for p in (base, local):
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if not isinstance(obj, dict):
            raise ValueError(f"设置文件格式错误（需要 mapping 类型）：{p}")
        merged = _deep_merge(merged, obj)
    return merged


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，b 中的值覆盖 a 中同键的值。"""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _flatten_settings_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    将嵌套的 YAML 配置展开为扁平的 Settings 字段。
    例如 {llm: {provider: ...}} → {llm_provider: ...}
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
        if "chunk_strategy" in rag:
            out["rag_chunk_strategy"] = str(rag["chunk_strategy"])
        if "chunk_size" in rag:
            out["rag_chunk_size"] = int(rag["chunk_size"])
        if "chunk_overlap" in rag:
            out["rag_chunk_overlap"] = int(rag["chunk_overlap"])
        if "similarity_threshold" in rag:
            out["rag_similarity_threshold"] = float(rag["similarity_threshold"])
        if "huggingface_hub_endpoint" in rag:
            v = rag.get("huggingface_hub_endpoint")
            out["huggingface_hub_endpoint"] = (str(v).strip() if v else None) or None

    return out
