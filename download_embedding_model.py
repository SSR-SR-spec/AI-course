"""
Download a sentence-transformers model to disk for offline use.

Run from the backend folder:
  python scripts/download_embedding_model.py
  python scripts/download_embedding_model.py sentence-transformers/all-MiniLM-L6-v2

Then set rag.embedding_model in app/settings.yaml to the printed absolute path
(forward slashes are OK on Windows).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure HF mirror from settings.yaml is applied before huggingface_hub imports.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import apply_hf_hub_endpoint_before_imports  # noqa: E402

apply_hf_hub_endpoint_before_imports()

from huggingface_hub import snapshot_download  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    safe_name = model_id.split("/")[-1].replace(" ", "_")
    out_dir = _BACKEND_ROOT / "models" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id!r} -> {out_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print("Done.\n")
    print("Set in app/settings.yaml under rag:")
    # YAML-friendly: forward slashes
    p = out_dir.resolve().as_posix()
    print(f'  embedding_model: "{p}"')


if __name__ == "__main__":
    main()
