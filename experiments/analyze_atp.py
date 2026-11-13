"""Compare the attribution-patching baseline with today's graph and Fix B on the same edits.

    python experiments/analyze_atp.py [--model gemma2]
    (needs results/fidelity_atp[_gemma2].parquet from atp_baseline.py)

Adds ``atp_baseline`` (GPT-2) or ``atp_baseline_gemma2`` to results/summary.json: per-predictor
metrics with 95% cluster-bootstrap intervals over prompts, and paired differences Fix B − AtP on
the same resampled edits.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import RESULTS, check_edit_keys, update_summary

from twinlens.fidelity.metrics import add_derived, cluster_bootstrap, level_metrics
from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS

LEVELS = ["direct", "linear", "linear_fulljac", "replacement_full", "atp"]
METRICS = ["spearman", "sign_agree", "within_tol_big", "catastrophic", "mean_err_clipped"]
SEED = 2  # levels first, then the paired difference
BOOT_LEVELS = 500
BOOT_PAIRED = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2", "gemma2"], default="gpt2")
    args = ap.parse_args()
    tag = "" if args.model == "gpt2" else f"_{args.model}"
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(RESULTS / f"fidelity{tag}.parquet")
    atp = pd.read_parquet(RESULTS / f"fidelity_atp{tag}.parquet")
    check_edit_keys(
        df, atp, f"fidelity_atp{tag}.parquet vs fidelity{tag}.parquet (re-run atp_baseline.py)"
    )
    df["atp_dlogp"] = atp.atp_dlogp.to_numpy()
    df["atp_cos"] = np.nan  # AtP predicts one logit, not the full Δlogits
    df = add_derived(df, [*TWIN_LEVELS, *(f for f in FIX_LEVELS if f"{f}_dlogp" in df), "atp"])

    levels = {
        lv: cluster_bootstrap(
            df, lambda d, lv=lv: level_metrics(d, lv, keys=METRICS), BOOT_LEVELS, rng
        )
        for lv in LEVELS
    }

    def paired(d: pd.DataFrame) -> dict[str, float]:
        fixed = level_metrics(d, "linear_fulljac", keys=METRICS)
        atp_m = level_metrics(d, "atp", keys=METRICS)
        return {k: fixed[k] - atp_m[k] for k in METRICS}

    paired_diff = cluster_bootstrap(df, paired, BOOT_PAIRED, rng)
    out = {"levels": levels, "fixB_minus_atp": paired_diff, "n_edits": len(df)}
    key = f"atp_baseline{tag}"
    meta = {
        key: {
            "seed": SEED,
            "n_boot_levels": BOOT_LEVELS,
            "n_boot_paired": BOOT_PAIRED,
            "unit": "prompt",
        }
    }
    update_summary({key: out}, meta=meta)
    for lv in LEVELS:
        print(f"{lv:18s}", {k: [round(x, 3) for x in v] for k, v in levels[lv].items()})
    print("Fix B - AtP", {k: [round(x, 3) for x in v] for k, v in paired_diff.items()})


if __name__ == "__main__":
    main()
