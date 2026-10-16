"""Analyse the fidelity sweep: ladder metrics, decay with depth, error sources, trust model.

    python experiments/analyze.py [--small]

Writes ``results/summary.json`` (consumed by the API/UI and the report),
``results/trust_model.json`` and figures under ``experiments/figures``.
All uncertainty is a 95% cluster bootstrap over *prompts* (edits within a prompt are not
independent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinlens.fidelity.signals import SIGNALS  # noqa: E402
from twinlens.fidelity.trust import TrustModel, auroc, fit_logistic, transform  # noqa: E402
from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "experiments" / "figures"
TRUST_LEVEL = "linear"  # the attribution-graph twin: what a graph's edges claim
CLIP = 5.0  # nats; errors are clipped here before averaging so one blow-up can't dominate
BOOT = 1000
RNG = np.random.default_rng(0)

LEVEL_LABELS = {
    "direct": "Direct edges",
    "linear": "Linear (graph)",
    "replacement": "Replacement",
    "replacement_ln": "+ live LayerNorm",
    "replacement_full": "+ live attention",
    "linear_lnjac": "Graph + LN Jacobian",
    "linear_fulljac": "Graph + LN & attention Jacobians",
}
# The proposed fix, compared against today's graph twin and the best nonlinear twin.
FIX_COMPARE = ["linear", "linear_lnjac", "linear_fulljac", "replacement_full"]


# ------------------------------------------------------------------- metrics
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    for lv in [*TWIN_LEVELS, *(f for f in FIX_LEVELS if f"{f}_dlogp" in df)]:
        err = (df[f"{lv}_dlogp"] - df.real_dlogp).abs()
        df[f"{lv}_err"] = err
        df[f"{lv}_ok"] = err <= 0.1 + 0.25 * df.real_dlogp.abs()
        df[f"{lv}_catastrophic"] = err > 1.0
    df["real_big"] = df.real_dlogp.abs() > 0.05
    return df


def level_metrics(d: pd.DataFrame, lv: str) -> dict[str, float]:
    big = d[d.real_big]
    return {
        "spearman": d[[f"{lv}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1],
        "median_cos": d[f"{lv}_cos"].median(),
        "sign_agree": ((big[f"{lv}_dlogp"] > 0) == (big.real_dlogp > 0)).mean(),
        "spearman_big": big[[f"{lv}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1],
        "within_tol_big": big[f"{lv}_ok"].mean(),
        "within_tol": d[f"{lv}_ok"].mean(),
        "catastrophic": d[f"{lv}_catastrophic"].mean(),
        "mean_err_clipped": d[f"{lv}_err"].clip(upper=CLIP).mean(),
    }


def bootstrap(df: pd.DataFrame, fn, n: int = BOOT) -> dict[str, list[float]]:
    """Cluster bootstrap over prompts: returns {metric: [point, lo, hi]}."""
    point = fn(df)
    groups = {k: g for k, g in df.groupby("prompt_id")}
    ids = list(groups)
    samples = []
    for _ in range(n):
        pick = RNG.choice(ids, size=len(ids), replace=True)
        samples.append(fn(pd.concat([groups[i] for i in pick], ignore_index=True)))
    out = {}
    for k, v in point.items():
        vals = np.array([s[k] for s in samples], dtype=float)
        out[k] = [float(v), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return out


# ---------------------------------------------------------------- trust model
def design(df: pd.DataFrame) -> torch.Tensor:
    cols = [[transform(s, v) for v in df[s]] for s in SIGNALS]
    return torch.tensor(np.array(cols, dtype=np.float32).T)


def trust_analysis(df: pd.DataFrame, level: str, n_folds: int = 5) -> tuple[dict, TrustModel]:
    """Grouped (by prompt) cross-validated AUROC of predicting ``{level}_ok`` from signals."""
    y = torch.tensor(df[f"{level}_ok"].to_numpy(), dtype=torch.float32)
    X = design(df)
    prompts = df.prompt_id.unique()
    RNG.shuffle(prompts)
    folds = np.array_split(prompts, n_folds)
    scores = torch.zeros(len(df))
    for fold in folds:
        test = torch.tensor(df.prompt_id.isin(fold).to_numpy())
        mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
        w, b = fit_logistic((X[~test] - mu) / sd, y[~test])
        scores[test] = torch.sigmoid(((X[test] - mu) / sd) @ w + b)
    cv_auc = auroc(scores, y)
    big = torch.tensor(df.real_big.to_numpy())
    cv_auc_big = auroc(scores[big], y[big])

    # Single-signal baselines (sign chosen in the direction that helps, on all data: optimistic
    # for the baselines, which makes the comparison conservative for the trust model).
    singles = {}
    for j, s in enumerate(SIGNALS):
        a = auroc(X[:, j], y)
        singles[s] = max(a, 1 - a)

    # Calibration on the cross-validated scores.
    bins = np.linspace(0, 1, 6)
    calib = []
    sc = scores.numpy()
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (sc >= lo) & (sc < hi) if hi < 1 else (sc >= lo)
        if m.sum() >= 10:
            calib.append(
                {"pred": float(sc[m].mean()), "obs": float(y.numpy()[m].mean()), "n": int(m.sum())}
            )

    # Precision of the trust score where it matters: of the edits flagged "trusted" (top half),
    # how many really were within tolerance, vs the base rate.
    order = scores.argsort(descending=True)
    top = order[: len(order) // 2]
    bottom = order[len(order) // 2 :]

    mu, sd = X.mean(0), X.std(0) + 1e-6
    w, b = fit_logistic((X - mu) / sd, y)
    model = TrustModel(
        features=list(SIGNALS),
        mean=mu.tolist(),
        std=sd.tolist(),
        coef=w.tolist(),
        intercept=b,
        target=f"P({level} twin within tolerance: |err| <= 0.1 + 0.25|real effect|)",
        metrics={"cv_auroc": cv_auc, "base_rate": float(y.mean())},
    )
    return {
        "level": level,
        "cv_auroc": cv_auc,
        "cv_auroc_big": cv_auc_big,
        "base_rate_big": float(y[big].mean()),
        "base_rate": float(y.mean()),
        "ok_rate_trusted_half": float(y[top].mean()),
        "ok_rate_flagged_half": float(y[bottom].mean()),
        "single_signal_auroc": singles,
        "coefficients": dict(zip(SIGNALS, w.tolist(), strict=True)),
        "calibration": calib,
        "_cv_scores": scores,
    }, model


def trust_transfer(df: pd.DataFrame, level: str) -> dict[str, float]:
    """Leave-one-task-family-out AUROC: train on 7 families, test on the unseen 8th."""
    y = torch.tensor(df[f"{level}_ok"].to_numpy(), dtype=torch.float32)
    X = design(df)
    out = {}
    for fam in sorted(df.category.unique()):
        test = torch.tensor((df.category == fam).to_numpy())
        mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
        w, b = fit_logistic((X[~test] - mu) / sd, y[~test])
        out[fam] = auroc(((X[test] - mu) / sd) @ w + b, y[test])
    out["mean"] = float(np.nanmean(list(out.values())))
    return out


def trusted_edges_payoff(df: pd.DataFrame, scores: torch.Tensor, level: str) -> dict:
    """On the edges a user would inspect (top attribution edges): how often is the graph right,
    overall vs. on the ones the trust model clears (CV scores >= 0.7), and what share is kept."""
    top = (df.selection == "top").to_numpy()
    ok = df[f"{level}_ok"].to_numpy()
    sc = scores.numpy()
    big = df.real_big.to_numpy()
    res = {}
    for name, m in [("all_top_edges", top), ("top_edges_big_effect", top & big)]:
        kept = m & (sc >= 0.7)
        res[name] = {
            "ok_all": float(ok[m].mean()),
            "ok_trusted": float(ok[kept].mean()),
            "ok_flagged": float(ok[m & (sc < 0.7)].mean()),
            "kept_share": float(kept.sum() / m.sum()),
            "n": int(m.sum()),
        }
    return res


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(RESULTS / ("fidelity_small.parquet" if args.small else "fidelity.parquet"))
    df = add_derived(df)
    n_prompts = df.prompt_id.nunique()
    print(f"{len(df)} edits over {n_prompts} prompts")

    ladder = {lv: bootstrap(df, lambda d, lv=lv: level_metrics(d, lv)) for lv in TWIN_LEVELS}

    # Fidelity vs depth: how many layers sit between the edit and the output.
    bins = [(0, 2), (3, 5), (6, 8), (9, 11)]
    decay = {}
    for lv in TWIN_LEVELS:
        decay[lv] = []
        for lo, hi in bins:
            d = df[(df.layers_after >= lo) & (df.layers_after <= hi)]
            decay[lv].append(
                {
                    "layers_after": f"{lo}-{hi}",
                    "median_cos": float(d[f"{lv}_cos"].median()),
                    "within_tol": float(d[f"{lv}_ok"].mean()),
                    "catastrophic": float(d[f"{lv}_catastrophic"].mean()),
                    "n": len(d),
                }
            )

    # Error decomposition along the ladder (clipped mean error, replacement = 100%).
    e = {lv: ladder[lv]["mean_err_clipped"][0] for lv in TWIN_LEVELS}
    total = e["replacement"]
    decomposition = {
        "frozen_layernorm": (e["replacement"] - e["replacement_ln"]) / total,
        "frozen_attention": (e["replacement_ln"] - e["replacement_full"]) / total,
        "transcoder_residual": e["replacement_full"] / total,
    }

    by = {}
    for col in ["selection", "kind", "category"]:
        by[col] = {
            str(k): {lv: level_metrics(g, lv) for lv in (TRUST_LEVEL, "replacement_full")}
            for k, g in df.groupby(col)
        }

    trust, model = trust_analysis(df, TRUST_LEVEL)
    trust_full, _ = trust_analysis(df, "replacement_full")
    model.save(RESULTS / "trust_model.json")
    cv_scores = trust.pop("_cv_scores")
    trust_full.pop("_cv_scores")
    trust["transfer_lofo"] = trust_transfer(df, TRUST_LEVEL)
    trust["payoff"] = trusted_edges_payoff(df, cv_scores, TRUST_LEVEL)

    fix_levels = [lv for lv in FIX_COMPARE if f"{lv}_dlogp" in df]
    fix = {
        "levels": fix_levels,
        "ladder": {lv: bootstrap(df, lambda d, lv=lv: level_metrics(d, lv)) for lv in fix_levels},
        "by_category_spearman": {
            str(c): {
                lv: float(g[[f"{lv}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1])
                for lv in fix_levels
            }
            for c, g in df.groupby("category")
        },
        "top_edges_catastrophic": {
            lv: float(df[df.selection == "top"][f"{lv}_catastrophic"].mean()) for lv in fix_levels
        },
    }

    summary = {
        "n_edits": len(df),
        "n_prompts": n_prompts,
        "levels": list(TWIN_LEVELS),
        "level_labels": LEVEL_LABELS,
        "ladder": ladder,
        "decay": decay,
        "decomposition": decomposition,
        "breakdown": by,
        "trust": trust,
        "trust_replacement_full": {k: trust_full[k] for k in ("cv_auroc", "base_rate")},
        "fix": fix,
        "real_effect": {
            "median_abs_dlogp": float(df.real_dlogp.abs().median()),
            "frac_big": float(df.real_big.mean()),
            "frac_flip": float(df.real_flip.mean()),
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(
        json.dumps(
            {lv: {k: round(v[0], 3) for k, v in m.items()} for lv, m in ladder.items()}, indent=1
        )
    )
    print("decomposition", {k: round(v, 3) for k, v in decomposition.items()})
    print(
        "trust",
        {
            k: trust[k]
            for k in (
                "cv_auroc",
                "base_rate",
                "cv_auroc_big",
                "base_rate_big",
                "ok_rate_trusted_half",
                "ok_rate_flagged_half",
            )
        },
    )
    print(
        "big-effect",
        {
            lv: (round(ladder[lv]["spearman_big"][0], 3), round(ladder[lv]["within_tol_big"][0], 3))
            for lv in TWIN_LEVELS
        },
    )
    print("single", {k: round(v, 3) for k, v in trust["single_signal_auroc"].items()})
    print(
        "transfer (leave-one-family-out)",
        {k: round(v, 3) for k, v in trust["transfer_lofo"].items()},
    )
    print("payoff", trust["payoff"])
    print("fix", {lv: {k: round(v[0], 3) for k, v in m.items()} for lv, m in fix["ladder"].items()})
    print("fix top-edge catastrophic", fix["top_edges_catastrophic"])

    from figures import make_figures  # local module, imported late so analysis runs headless

    make_figures(summary, df, FIGS)


if __name__ == "__main__":
    main()
