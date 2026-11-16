"""Held-out external prompts (plan 2.1-2.2; pre-registered in DECISIONS.md D9 before the sweep).

    python experiments/analyze_external.py --tag gpt2|gemma2|gemma2_clt

Needs results/fidelity[_<model>]_ext.parquet (fidelity_sweep.py --prompts external). Adds
``external_<tag>`` to results/summary.json: the metrics of analyze_clt.py per level, the paired
Fix B - linear differences, the same for the contrast target (change of logit(answer) -
logit(contrast)), and all of it per family. Intervals are 95% cluster bootstraps over prompts.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import RESULTS, update_summary
from scipy.stats import spearmanr

from twinlens.fidelity.metrics import add_derived, cluster_bootstrap, level_metrics

METRICS = ["spearman", "sign_agree", "within_tol_big", "catastrophic", "mean_err_clipped"]
FIX = ("linear_fulljac", "linear")
MOVES = 0.1  # nat: an edit "moves" the logit difference when |real Δ| exceeds this
SEED, BOOT = 0, 1000
FILES = {
    "gpt2": "fidelity_ext.parquet",
    "gemma2": "fidelity_gemma2_ext.parquet",
    "gemma2_clt": "fidelity_gemma2_clt_ext.parquet",
}


def _levels(df: pd.DataFrame) -> list[str]:
    order = [
        "direct",
        "linear",
        "linear_lnjac",
        "linear_fulljac",
        "replacement",
        "replacement_ln",
        "replacement_full",
    ]
    return [lv for lv in order if f"{lv}_dlogp" in df]


def _dld(d: pd.DataFrame, lv: str) -> dict[str, float]:
    moves = d.real_dld.abs() > MOVES
    rho = spearmanr(d[f"{lv}_dld"], d.real_dld).statistic if len(d) > 2 else np.nan
    sign = ((d.loc[moves, f"{lv}_dld"] > 0) == (d.loc[moves, "real_dld"] > 0)).mean()
    return {"dld_spearman": float(rho), "dld_sign_agree": float(sign)}


def _all(d: pd.DataFrame, lv: str) -> dict[str, float]:
    return level_metrics(d, lv, METRICS) | _dld(d, lv)


def _block(d: pd.DataFrame, levels: list[str], rng: np.random.Generator) -> dict:
    out = {
        "levels": {
            lv: cluster_bootstrap(d, lambda x, lv=lv: _all(x, lv), BOOT, rng) for lv in levels
        }
    }

    def paired(x: pd.DataFrame) -> dict[str, float]:
        a, b = _all(x, FIX[0]), _all(x, FIX[1])
        return {m: a[m] - b[m] for m in a}

    out["fix_minus_linear"] = cluster_bootstrap(d, paired, BOOT, rng)
    out["n_edits"], out["n_prompts"] = len(d), int(d.prompt_id.nunique())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=list(FILES), required=True)
    args = ap.parse_args()
    df = pd.read_parquet(RESULTS / FILES[args.tag])
    levels = _levels(df)
    d = add_derived(df, levels)
    rng = np.random.default_rng(SEED)
    res = {"all": _block(d, levels, rng)}
    res["by_family"] = {fam: _block(g, levels, rng) for fam, g in d.groupby("category")}
    key = f"external_{args.tag}"
    update_summary(
        {key: res}, meta={key: {"seed": SEED, "n_boot": BOOT, "unit": "prompt", "decisions": "D9"}}
    )
    a = res["all"]
    print(
        f"{key}: {a['n_prompts']} prompts, {a['n_edits']} edits; rho linear "
        f"{a['levels']['linear']['spearman']}, fix {a['levels']['linear_fulljac']['spearman']}, "
        f"fix - linear {a['fix_minus_linear']['spearman']}"
    )


if __name__ == "__main__":
    main()
