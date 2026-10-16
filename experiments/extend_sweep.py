"""Add new twin levels to an existing sweep without re-running the old ones.

    python experiments/extend_sweep.py --modes linear_lnjac [--max-minutes 8]

For every prompt, the edits are re-sampled with the same seed and checked against the stored
rows (so the new columns line up with exactly the same interventions). The baseline "linear"
level is recomputed too and must match the stored values, as a regression check on the code.
Results go to ``experiments/results/raw_ext_<modes>/<id>.parquet``; ``fidelity.parquet`` is rebuilt
with the extra columns. Resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinlens.core.gpt2 import GPT2  # noqa: E402
from twinlens.fidelity.engine import _effect_metrics, sample_edits  # noqa: E402
from twinlens.twin.replacement import Twin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["linear_lnjac"])
    ap.add_argument("--n-top", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--max-minutes", type=float, default=0)
    args = ap.parse_args()

    corpus = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    prompts = [(cat, p) for cat, ps in corpus.items() for p in ps]
    out = RESULTS / ("raw_ext_" + "_".join(args.modes))
    out.mkdir(parents=True, exist_ok=True)
    twin = Twin(GPT2.load("gpt2"))
    t0 = time.perf_counter()
    worst_regression = 0.0

    for i, (_cat, text) in enumerate(prompts):
        path = out / f"{i:04d}.parquet"
        if path.exists():
            continue
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        t = time.perf_counter()
        old = pd.read_parquet(RESULTS / "raw" / f"{i:04d}.parquet")
        trace = twin.trace(twin.model.to_tokens(text))
        clean = trace.logits[-1]
        target = int(clean.argmax())
        effects = twin.feature_effects(trace, target)
        specs = sample_edits(twin, trace, effects, args.n_top, args.n_random, seed=i)
        key_new = [(s.edit.layer, s.edit.pos, s.edit.feature, s.kind) for s in specs]
        key_old = list(zip(old.layer.astype(int), old.pos, old.feature, old.kind, strict=True))
        assert key_new == key_old, f"prompt {i}: re-sampled edits differ from the stored sweep"

        cols: dict[str, list] = {}
        for start in range(0, len(specs), args.chunk):
            edits = [s.edit for s in specs[start : start + args.chunk]]
            real = _effect_metrics(clean, twin.run(trace, "real", edits), target)
            for mode in ["linear", *args.modes]:
                m = _effect_metrics(clean, twin.run(trace, mode, edits), target)
                cos = torch.nn.functional.cosine_similarity(m["dlogits"], real["dlogits"], dim=-1)
                cols.setdefault(f"{mode}_dlogp", []).extend(m["dlogp"].tolist())
                cols.setdefault(f"{mode}_kl", []).extend(m["kl"].tolist())
                cols.setdefault(f"{mode}_flip", []).extend((m["top1"] != target).tolist())
                cols.setdefault(f"{mode}_cos", []).extend(cos.tolist())
        diff = (pd.Series(cols["linear_dlogp"]) - old.linear_dlogp.reset_index(drop=True)).abs()
        rel = float((diff / (old.linear_dlogp.abs().reset_index(drop=True) + 1e-2)).max())
        worst_regression = max(worst_regression, rel)
        new = pd.DataFrame({k: v for k, v in cols.items() if k.rsplit("_", 1)[0] in args.modes})
        new.to_parquet(path)
        print(
            f"[{i + 1}/{len(prompts)}] {time.perf_counter() - t:.0f}s  "
            f"linear regression (max rel) {rel:.1e}",
            flush=True,
        )

    base = pd.read_parquet(RESULTS / "fidelity.parquet")
    ext = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))], ignore_index=True)
    assert len(ext) == len(base)
    for c in ext.columns:
        base[c] = ext[c].to_numpy()
    base.to_parquet(RESULTS / "fidelity.parquet")
    print(
        f"added {list(ext.columns)} to fidelity.parquet; "
        f"worst linear regression {worst_regression:.1e}"
    )


if __name__ == "__main__":
    main()
