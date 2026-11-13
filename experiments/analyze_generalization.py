"""Does the fix generalise? Paired comparison of today's graph twin vs. the Jacobian fix on
GPT-2 small (8,064 edits), Qwen3-0.6B (2,304 edits, layers 10-27) and Gemma-2-2B (when
results/fidelity_gemma2.parquet exists).

    python experiments/analyze_generalization.py

Adds ``generalization`` to results/summary.json and draws figures/F7_generalization.
Differences are paired (same edits), with 95% cluster bootstraps over prompts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import FIGS, RESULTS, update_summary

from twinlens.fidelity.metrics import (
    CLIP,
    add_derived,
    cluster_bootstrap,
    level_metrics,
    within_tol,
)

LEVELS = ["direct", "linear", "linear_lnjac", "linear_fulljac"]
METRICS = ["spearman", "sign_agree", "within_tol_big", "catastrophic", "mean_err_clipped"]
MODELS = {
    "gpt2": ("GPT-2 small", "fidelity.parquet"),
    "qwen3": ("Qwen3-0.6B", "fidelity_qwen3.parquet"),
    "gemma2": ("Gemma-2-2B", "fidelity_gemma2.parquet"),
}
# one generator for all models, in this order (levels, then the paired difference); a model
# appended at the end leaves the earlier models' intervals unchanged
SEED = 0
BOOT_LEVELS = 500
BOOT_PAIRED = 1000


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {}
    for key, (label, file) in MODELS.items():
        if not (RESULTS / file).exists():
            print(f"{label}: {file} missing, skipped")
            continue
        df = add_derived(pd.read_parquet(RESULTS / file), LEVELS)
        levels = {
            lv: cluster_bootstrap(
                df, lambda d, lv=lv: level_metrics(d, lv, keys=METRICS), BOOT_LEVELS, rng
            )
            for lv in LEVELS
        }

        def paired(d: pd.DataFrame) -> dict[str, float]:
            fixed = level_metrics(d, "linear_fulljac", keys=METRICS)
            graph = level_metrics(d, "linear", keys=METRICS)
            return {m: fixed[m] - graph[m] for m in METRICS}

        diff = cluster_bootstrap(df, paired, BOOT_PAIRED, rng)
        fam = {
            str(c): {
                lv: float(g[[f"{lv}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1])
                for lv in ("linear", "linear_fulljac")
            }
            for c, g in df.groupby("category")
        }
        zero_err = df[df.real_big].real_dlogp.abs()  # predicting "no change" has error |real|
        out[key] = {
            "label": label,
            "n_edits": len(df),
            "n_prompts": int(df.prompt_id.nunique()),
            "layers": sorted(int(x) for x in df.layer.unique()),
            "levels": levels,
            "paired_diff": diff,
            "by_category_spearman": fam,
            "no_change": {
                "within_tol_big": float(within_tol(err=zero_err, real=zero_err).mean()),
                "mean_err_clipped": float(df.real_dlogp.abs().clip(upper=CLIP).mean()),
                "n_big": int(df.real_big.sum()),
            },
        }
        print(label, {m: [round(v, 3) for v in diff[m]] for m in METRICS})
    meta = {
        "generalization": {
            "seed": SEED,
            "n_boot_levels": BOOT_LEVELS,
            "n_boot_paired": BOOT_PAIRED,
            "unit": "prompt",
        }
    }
    update_summary({"generalization": out}, meta=meta)

    from figures import generalization_figure

    generalization_figure(out, FIGS)


if __name__ == "__main__":
    main()
