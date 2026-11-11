"""Paths and I/O helpers shared by the experiment scripts.

Importing this module puts the repository root on ``sys.path``, so scripts stay runnable as
``python experiments/<script>.py`` and can import ``twinlens`` / ``server`` without installing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
FIGS = EXPERIMENTS / "figures"
SUMMARY = RESULTS / "summary.json"

# Identifies one edit; positional merges between sweep files are checked against it.
EDIT_KEY = ("prompt_id", "layer", "pos", "feature", "kind")


def _replace_atomically(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """``df.to_parquet(path)`` via a temp file + rename, so a crash never leaves a partial file."""
    _replace_atomically(Path(path), df.to_parquet)


def update_summary(
    sections: dict, meta: dict | None = None, path: Path = SUMMARY
) -> dict:
    """Replace only the given top-level keys of summary.json (other scripts' sections survive).

    ``meta`` entries (e.g. ``{"ladder": {"seed": 0, "n_boot": 1000}}``) are merged into the
    ``meta`` block by section name. The file is written atomically. Returns the new summary.
    """
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update(sections)
    if meta:
        summary["meta"] = {**summary.get("meta", {}), **meta}
    text = json.dumps(summary, indent=2, default=float)
    _replace_atomically(path, lambda p: p.write_text(text))
    return summary


def check_edit_keys(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    """Fail unless ``a`` and ``b`` list the same edits in the same row order."""
    if len(a) != len(b):
        raise ValueError(f"{what}: {len(a)} vs {len(b)} rows")
    missing = [c for c in EDIT_KEY if c not in a or c not in b]
    if missing:
        raise ValueError(f"{what}: key columns {missing} missing; cannot check row alignment")
    bad = np.zeros(len(a), dtype=bool)
    for c in EDIT_KEY:
        bad |= a[c].to_numpy() != b[c].to_numpy()
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{what}: {int(bad.sum())} rows misaligned, first at row {i}: "
            f"{a[list(EDIT_KEY)].iloc[i].tolist()} vs {b[list(EDIT_KEY)].iloc[i].tolist()}"
        )
