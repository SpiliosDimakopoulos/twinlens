"""TwinLens: a digital twin of GPT-2's internals that measures how far its explanations hold."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

# numpy is only used for memory-mapping; its OpenBLAS thread pool would otherwise reserve
# per-thread buffers that matter on low-memory machines. torch keeps its own thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__all__ = ["GPT2", "FeatureEdit", "Qwen3", "TrustModel", "Twin", "audit"]

# Imported on first access, so `import twinlens` (and `twinlens --help`) does not load torch.
_LAZY = {
    "GPT2": "twinlens.core.gpt2",
    "Qwen3": "twinlens.core.qwen3",
    "Twin": "twinlens.twin.replacement",
    "FeatureEdit": "twinlens.twin.replacement",
    "TrustModel": "twinlens.fidelity.trust",
    "audit": "twinlens.cli",
}

if TYPE_CHECKING:
    from twinlens.cli import audit
    from twinlens.core.gpt2 import GPT2
    from twinlens.core.qwen3 import Qwen3
    from twinlens.fidelity.trust import TrustModel
    from twinlens.twin.replacement import FeatureEdit, Twin


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'twinlens' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
