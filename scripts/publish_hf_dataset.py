"""Publish the TwinLens fidelity parquets to the Hugging Face Hub.

Usage:
    HF_TOKEN=... python scripts/publish_hf_dataset.py
or log in first with `huggingface-cli login`.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "SpiliosDimakopoulos/twinlens-fidelity"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FILES = ["fidelity.parquet", "fidelity_qwen3.parquet"]
CARD = ROOT / "docs" / "HF_DATASET_CARD.md"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=token)
    for name in FILES:
        path = RESULTS / name
        if not path.exists():
            raise FileNotFoundError(path)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {name}",
        )
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"Published: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
