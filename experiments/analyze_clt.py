"""CLT pilot analysis (plan 0.8; pre-registered in DECISIONS.md D7 before the sweep ran).

    python experiments/analyze_clt.py [--small]

Needs results/fidelity_gemma2_clt.parquet (fidelity_sweep.py --transcoders clt426k), optionally
results/fidelity_atp_gemma2_clt.parquet (atp_baseline.py --transcoders clt426k) and, for the
comparison with per-layer transcoders, results/fidelity_gemma2.parquet. Adds ``gemma2_clt`` to
results/summary.json. Intervals are 95% cluster bootstraps over prompts.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import RESULTS, check_edit_keys, update_summary

from twinlens.fidelity.metrics import add_derived, cluster_bootstrap, level_metrics

TWIN = ["direct", "linear", "linear_lnjac", "linear_fulljac"]
METRICS = ["spearman", "sign_agree", "within_tol_big", "catastrophic", "mean_err_clipped"]
SEED = 0
BOOT = 1000
GATE_FIX = ("linear_fulljac", "linear")


def _with_truth(df: pd.DataFrame, truth: str, levels: list[str]) -> pd.DataFrame:
    """A copy whose ``real_*`` columns are the chosen ground truth (``real`` or
    ``real_frozen_attn``), with the derived error columns recomputed."""
    d = df.copy()
    if truth != "real":
        d["real_dlogp"] = d[f"{truth}_dlogp"]
    return add_derived(d, levels)


def _block(d: pd.DataFrame, levels: list[str], rng: np.random.Generator) -> dict:
    out = {
        "levels": {
            lv: cluster_bootstrap(d, lambda x, lv=lv: level_metrics(x, lv, METRICS), BOOT, rng)
            for lv in levels
        }
    }

    def paired(a: str, b: str):
        def fn(x: pd.DataFrame) -> dict[str, float]:
            ma, mb = level_metrics(x, a, METRICS), level_metrics(x, b, METRICS)
            return {m: ma[m] - mb[m] for m in METRICS}

        return fn

    out["fix_minus_linear"] = cluster_bootstrap(d, paired(*GATE_FIX), BOOT, rng)
    if "atp" in levels:
        out["fix_minus_atp"] = cluster_bootstrap(d, paired("linear_fulljac", "atp"), BOOT, rng)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    small = "_small" if args.small else ""
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(RESULTS / f"fidelity{small}_gemma2_clt.parquet")
    levels = list(TWIN)
    atp_path = RESULTS / f"fidelity_atp{small}_gemma2_clt.parquet"
    if atp_path.exists():
        atp = pd.read_parquet(atp_path)
        check_edit_keys(df, atp, f"{atp_path.name} vs the CLT sweep")
        df["atp_dlogp"] = atp.atp_dlogp.to_numpy()
        df["atp_cos"] = np.nan
        levels.append("atp")

    res: dict = {"n_edits": len(df), "n_prompts": int(df.prompt_id.nunique())}
    # primary ground truth: the real model, everything live (as for every other number)
    d = _with_truth(df, "real", [*levels, "real_frozen_attn"])
    res["vs_real"] = _block(d, levels, rng)
    # circuit-tracer's default intervention as a second ground truth
    if "real_frozen_attn_dlogp" in df:
        res["vs_real_frozen_attn"] = _block(
            _with_truth(df, "real_frozen_attn", levels), levels, rng
        )
        res["frozen_attn_vs_real"] = level_metrics(d, "real_frozen_attn", METRICS)

    res["by_layer_spearman"] = {
        int(l): {lv: float(level_metrics(g, lv, ["spearman"])["spearman"]) for lv in levels}
        | {"n": len(g)}
        for l, g in d.groupby("layer")
    }

    # PLT on the same prompts: joint prompt bootstrap of (CLT linear - PLT Fix B)
    plt_path = RESULTS / f"fidelity{small}_gemma2.parquet"
    gate_gap = None
    if plt_path.exists():
        plt = pd.read_parquet(plt_path)
        plt = add_derived(plt[plt.prompt_id.isin(df.prompt_id.unique())].copy(), TWIN)
        both = pd.concat([d.assign(tc="clt"), plt.assign(tc="plt")], ignore_index=True)

        def gap(x: pd.DataFrame) -> dict[str, float]:
            c = level_metrics(x[x.tc == "clt"], "linear", METRICS)
            p = level_metrics(x[x.tc == "plt"], "linear_fulljac", METRICS)
            return {m: c[m] - p[m] for m in METRICS}

        res["plt_same_prompts"] = {lv: level_metrics(plt, lv, METRICS) for lv in TWIN} | {
            "n_edits": len(plt)
        }
        res["clt_linear_minus_plt_fix"] = cluster_bootstrap(both, gap, BOOT, rng)
        gate_gap = res["clt_linear_minus_plt_fix"]["spearman"]

    # the decision gate of plan 0.9, as pre-registered in DECISIONS.md D7
    _, lo, _ = res["vs_real"]["fix_minus_linear"]["spearman"]
    if gate_gap is None:
        gate = "undecided (no PLT results)"
    elif lo > 0 and gate_gap[2] < 0:
        gate = "A"
    elif lo <= 0 and gate_gap[2] >= 0:
        gate = "B"
    else:
        gate = "C"
    res["gate"] = gate
    key = "gemma2_clt" + ("_small" if args.small else "")
    update_summary(
        {key: res},
        meta={key: {"seed": SEED, "n_boot": BOOT, "unit": "prompt", "decisions": "D7"}},
    )
    print(f"gate {gate}; fix - linear spearman {res['vs_real']['fix_minus_linear']['spearman']}")


if __name__ == "__main__":
    main()
