"""Feature-to-feature edges: analysis fixed in DECISIONS.md D8 (plan 1.5).

    python experiments/analyze_edges.py --tag gpt2 [--small]   # or gemma2, gemma2_clt

Reads results/edges[_small]_<tag>.parquet (edge_sweep.py) and writes ``edges_<tag>`` to
results/summary.json. Intervals are 95% cluster bootstraps over prompts (seed 0, 1000
resamples), computed on numpy arrays because the tables have one row per (edit, target).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import RESULTS, update_summary
from scipy.stats import rankdata

LEVELS = ("linear", "linear_lnjac", "linear_fulljac")
TRUTHS = ("real", "real_frozen_attn")
FIX = ("linear_fulljac", "linear")
MOVES = 0.1  # a target "really moves" when |real Δact| > MOVES * act0
DIST_BINS = {"1": (1, 1), "2-3": (2, 3), "4-7": (4, 7), "8+": (8, 99)}
SEED, BOOT = 0, 1000


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _metrics(c: dict[str, np.ndarray], truth: str) -> dict[str, float]:
    """All metrics on one (re)sample ``c`` of column arrays."""
    t, act0 = c[f"{truth}_dact"], c["act0"]
    moves = np.abs(t) > MOVES * act0
    gate = c[f"{truth}_dpre"] - t  # non-zero only where the target turned off
    out = {"turned_off": float(np.mean(np.isclose(t, -act0))), "n": float(len(t))}
    for lv in LEVELS:
        p = c[f"{lv}_dact"]
        sq = np.sum((p - t) ** 2)
        out |= {
            f"{lv}/spearman": _rho(p, t),
            f"{lv}/spearman_rel": _rho(p / act0, t / act0),
            f"{lv}/sign_agree": float(np.mean((p[moves] > 0) == (t[moves] > 0))),
            f"{lv}/gate_share_of_sq_err": float(np.sum(gate**2) / sq) if sq > 0 else float("nan"),
        }
    for m in ("spearman", "spearman_rel", "sign_agree"):
        out[f"fix_minus_linear/{m}"] = out[f"{FIX[0]}/{m}"] - out[f"{FIX[1]}/{m}"]
    return out


def _boot(d: pd.DataFrame, truth: str, rng: np.random.Generator) -> dict[str, list[float]]:
    cols = ["act0", *(f"{lv}_dact" for lv in LEVELS), f"{truth}_dact", f"{truth}_dpre"]
    groups = {k: {c: g[c].to_numpy() for c in cols} for k, g in d.groupby("prompt_id")}
    ids = list(groups)
    point = _metrics({c: d[c].to_numpy() for c in cols}, truth)
    samples = []
    for _ in range(BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        samples.append(
            _metrics({c: np.concatenate([groups[i][c] for i in pick]) for c in cols}, truth)
        )
    res: dict[str, list[float]] = {}
    for k, v in point.items():
        vals = np.array([s[k] for s in samples], dtype=float)
        res[k] = [v, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    small = "_small" if args.small else ""
    d = pd.read_parquet(RESULTS / f"edges{small}_{args.tag}.parquet")
    d["dist"] = d.target_layer - d.layer
    rng = np.random.default_rng(SEED)
    res: dict = {"n_pairs": len(d), "n_prompts": int(d.prompt_id.nunique())}
    for truth in TRUTHS:
        if f"{truth}_dact" not in d:
            continue
        block = {"all": _boot(d, truth, rng)}
        for sel, g in d.groupby("target_selection"):
            block[f"selection={sel}"] = _boot(g, truth, rng)
        for name, (lo, hi) in DIST_BINS.items():
            g = d[(d.dist >= lo) & (d.dist <= hi)]
            if g.prompt_id.nunique() > 1:
                block[f"dist={name}"] = _boot(g, truth, rng)
        res[f"vs_{truth}"] = block
    key = f"edges{small}_{args.tag}"
    update_summary(
        {key: res}, meta={key: {"seed": SEED, "n_boot": BOOT, "unit": "prompt", "decisions": "D8"}}
    )
    a = res["vs_real"]["all"]
    print(
        f"{key}: rho linear {a['linear/spearman']}, fix {a['linear_fulljac/spearman']}, "
        f"fix - linear {a['fix_minus_linear/spearman']}"
    )


if __name__ == "__main__":
    main()
