"""Does the fix generalise? Paired comparison of today's graph twin vs. the Jacobian fix on
GPT-2 small (8,064 edits) and Qwen3-0.6B (2,304 edits, layers 10-27).

    python experiments/analyze_generalization.py

Adds ``generalization`` to results/summary.json and draws figures/F7_generalization.
Differences are paired (same edits), with 95% cluster bootstraps over prompts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from analyze import level_metrics  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
LEVELS = ["direct", "linear", "linear_lnjac", "linear_fulljac"]
METRICS = ["spearman", "sign_agree", "within_tol_big", "catastrophic", "mean_err_clipped"]
MODELS = {
    "gpt2": ("GPT-2 small", "fidelity.parquet"),
    "qwen3": ("Qwen3-0.6B", "fidelity_qwen3.parquet"),
}
RNG = np.random.default_rng(0)


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / name)
    for lv in LEVELS:
        err = (df[f"{lv}_dlogp"] - df.real_dlogp).abs()
        df[f"{lv}_err"], df[f"{lv}_ok"], df[f"{lv}_catastrophic"] = (
            err,
            err <= 0.1 + 0.25 * df.real_dlogp.abs(),
            err > 1,
        )
    df["real_big"] = df.real_dlogp.abs() > 0.05
    return df


def boot(df: pd.DataFrame, fn, n: int = 1000) -> dict[str, list[float]]:
    groups = {k: g for k, g in df.groupby("prompt_id")}
    ids = list(groups)
    point = fn(df)
    samples = [fn(pd.concat([groups[i] for i in RNG.choice(ids, len(ids))])) for _ in range(n)]
    return {
        k: [float(point[k]), *map(float, np.percentile([s[k] for s in samples], [2.5, 97.5]))]
        for k in point
    }


def main() -> None:
    out = {}
    for key, (label, file) in MODELS.items():
        df = load(file)
        levels = {
            lv: boot(df, lambda d, lv=lv: {m: level_metrics(d, lv)[m] for m in METRICS}, 500)
            for lv in LEVELS
        }
        diff = boot(
            df,
            lambda d: {
                m: level_metrics(d, "linear_fulljac")[m] - level_metrics(d, "linear")[m]
                for m in METRICS
            },
        )
        fam = {
            str(c): {
                lv: float(g[[f"{lv}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1])
                for lv in ("linear", "linear_fulljac")
            }
            for c, g in df.groupby("category")
        }
        out[key] = {
            "label": label,
            "n_edits": len(df),
            "n_prompts": int(df.prompt_id.nunique()),
            "layers": sorted(int(x) for x in df.layer.unique()),
            "levels": levels,
            "paired_diff": diff,
            "by_category_spearman": fam,
        }
        print(label, {m: [round(v, 3) for v in diff[m]] for m in METRICS})
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["generalization"] = out
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    from figures import generalization_figure

    generalization_figure(out, ROOT / "experiments" / "figures")


if __name__ == "__main__":
    main()
