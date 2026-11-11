"""Shared process state: the lazily loaded twin, the trust model and the model lock."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from twinlens.core.gpt2 import GPT2
from twinlens.fidelity.trust import TrustModel
from twinlens.twin.replacement import Twin

ROOT = Path(__file__).resolve().parents[1]
# The model holds shared hook state, so every request that touches it holds LOCK.
LOCK = threading.Lock()
_LOAD = threading.Lock()
_TWIN: Twin | None = None


def get_twin() -> Twin:
    # A lock (not lru_cache) so concurrent first requests load the 0.5 GB model only once.
    global _TWIN
    if _TWIN is None:
        with _LOAD:
            if _TWIN is None:
                _TWIN = Twin(GPT2.load("gpt2"))
    return _TWIN


@lru_cache(maxsize=1)
def get_trust() -> TrustModel | None:
    path = ROOT / "experiments" / "results" / "trust_model.json"
    return TrustModel.load(path) if path.exists() else None


@contextmanager
def model_session() -> Iterator[Twin]:
    """Yield the twin while holding ``LOCK``; the only way request code touches the model."""
    twin = get_twin()
    with LOCK:
        yield twin
