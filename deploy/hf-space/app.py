"""Hugging Face Space entry point for TwinLens (Gradio SDK, free CPU tier).

Docker Spaces need a paid plan, so this runs the FastAPI app from a Gradio-SDK Space. On start
it fetches the source at a pinned commit and the prebuilt dashboard from the GitHub release,
then serves both with uvicorn on port 7860. GPT-2 and its transcoders (~2.3 GB) download from
the Hub on the first request after the Space wakes.
"""

import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "SpiliosDimakopoulos/twinlens"
REF = "d34d9d77e4213f36fdcf2310267110ccb2799057"
DIST = f"https://github.com/{REPO}/releases/download/v1.2.0/twinlens-web-dist.tar.gz"
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def fetch(url: str) -> tarfile.TarFile:
    with urllib.request.urlopen(url, timeout=120) as r:
        return tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")


if not (SRC / "server").exists():
    with fetch(f"https://codeload.github.com/{REPO}/tar.gz/{REF}") as tar:
        top = next(n.split("/")[0] for n in tar.getnames() if "/" in n)  # skips pax_global_header
        members = [m for m in tar.getmembers() if m.name.startswith(top + "/")]
        for m in members:
            m.name = m.name[len(top) + 1 :]
        tar.extractall(SRC, members=[m for m in members if m.name], filter="data")
if not (SRC / "web" / "dist" / "index.html").exists():
    with fetch(DIST) as tar:
        tar.extractall(SRC / "web" / "dist", filter="data")

host = os.environ.get("SPACE_HOST", "spiliosdmk-twinlens.hf.space")
os.environ.setdefault("TWINLENS_WS_ORIGINS", f"https://{host}")
sys.path.insert(0, str(SRC))

import uvicorn  # noqa: E402

from server.app import app  # noqa: E402

uvicorn.run(app, host="0.0.0.0", port=7860)
