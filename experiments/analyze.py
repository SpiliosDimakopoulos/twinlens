"""Analyse the fidelity sweep: ladder metrics, decay with depth, error sources, trust model.

    python experiments/analyze.py [--small]

Updates its sections of ``results/summary.json`` (consumed by the API/UI and the report; the
``generalization`` and ``atp_baseline`` sections written by the other scripts are kept), writes
``results/trust_model.json`` and figures under ``experiments/figures``.
All uncertainty is a 95% cluster bootstrap over *prompts* (edits within a prompt are not
independent).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from common import FIGS, RESULTS, ROOT, update_summary

from twinlens.fidelity.metrics import (
    CLIP,
    add_derived,
    cluster_bootstrap,
    level_metrics,
    within_tol,
)
from twinlens.fidelity.signals import SIGNALS
from twinlens.fidelity.trust import TrustModel, auroc, fit_logistic, transform
from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS

TRUST_LEVEL = "linear"  # the attribution-graph twin: what a graph's edges claim
BOOT = 1000
SEED = 0  # one generator, shared in call order by the ladder/fix bootstraps and the CV folds
CV_FOLDS = 5
ROBUST_SEED = 1  # robustness() has its own generator, so it leaves every other interval as is
ROBUST_BOOT = 300

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


# ---------------------------------------------------------------- trust model
def design(df: pd.DataFrame) -> torch.Tensor:
    cols = [[transform(s, v) for v in df[s]] for s in SIGNALS]
    return torch.tensor(np.array(cols, dtype=np.float32).T)


def trust_analysis(
    df: pd.DataFrame, level: str, rng: np.random.Generator, n_folds: int = CV_FOLDS
) -> tuple[dict, TrustModel]:
    """Grouped (by prompt) cross-validated AUROC of predicting ``{level}_ok`` from signals.
    ``rng`` shuffles the prompts into folds."""
    y = torch.tensor(df[f"{level}_ok"].to_numpy(), dtype=torch.float32)
    X = design(df)
    prompts = df.prompt_id.unique()
    rng.shuffle(prompts)
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


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    a = auroc(torch.tensor(score, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return float(max(a, 1 - a))


def robustness(df: pd.DataFrame, cv_scores: torch.Tensor, level: str) -> dict:
    """Checks the report quotes alongside the headline numbers: a "no change" baseline, simple
    failure signals the trust model must beat, threshold and clip sensitivity, layer-0 share of
    catastrophic misses, and a bootstrap interval on the trust model's AUROC. Uses its own RNG
    so adding it leaves every other interval in summary.json unchanged."""
    rng = np.random.default_rng(ROBUST_SEED)
    big, top = df.real_big.to_numpy(), (df.selection == "top").to_numpy()
    y = df[f"{level}_ok"].to_numpy().astype(float)
    real = df.real_dlogp.abs()
    # Predicting "no change" (zero) has error |real| on every edit.
    zero_ok = within_tol(err=real, real=real).to_numpy()
    sc = cv_scores.numpy()
    pred = df[f"{level}_dlogp"].abs().to_numpy()
    out: dict = {
        "no_change": {
            "within_tol": float(zero_ok.mean()),
            "within_tol_big": float(zero_ok[big].mean()),
            "mean_err_clipped": float(real.clip(upper=CLIP).mean()),
        }
    }
    signals = {"abs_graph_prediction": pred}
    if "linear_fulljac_dlogp" in df:
        signals["graph_vs_fix_disagreement"] = (
            (df[f"{level}_dlogp"] - df.linear_fulljac_dlogp).abs().to_numpy()
        )
    out["baseline_auroc"] = {
        k: {"all": _auc(v, y), "big": _auc(v[big], y[big])} for k, v in signals.items()
    }

    # Trust-model AUROC with a cluster bootstrap over prompts (CV scores held fixed).
    pid = df.prompt_id.to_numpy()
    prompts = np.unique(pid)
    idx = {p: np.flatnonzero(pid == p) for p in prompts}
    reps = {"all": [], "big": []}
    for _ in range(ROBUST_BOOT):
        rows = np.concatenate([idx[p] for p in rng.choice(prompts, len(prompts), replace=True)])
        reps["all"].append(_auc(sc[rows], y[rows]))
        b = rows[big[rows]]
        reps["big"].append(_auc(sc[b], y[b]))
    out["trust_auroc_ci"] = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in reps.items()
    }

    # Leave-one-family-out AUROC restricted to edits with a noticeable effect.
    X = design(df)
    yt = torch.tensor(y, dtype=torch.float32)
    lofo = {}
    for fam in sorted(df.category.unique()):
        test = (df.category == fam).to_numpy()
        tt = torch.tensor(test)
        mu, sd = X[~tt].mean(0), X[~tt].std(0) + 1e-6
        w, b = fit_logistic((X[~tt] - mu) / sd, yt[~tt])
        s = (((X[tt] - mu) / sd) @ w + b).numpy()
        m = big[test]
        lofo[fam] = float(auroc(torch.tensor(s[m]), yt[tt][torch.tensor(m)]))
    lofo["mean"] = float(np.mean(list(lofo.values())))
    out["transfer_lofo_big"] = lofo

    # Payoff on the top edges with a noticeable effect, across trust cut-offs, and the same
    # share kept by the simplest rule (smallest |graph prediction|).
    m = top & big
    out["payoff_thresholds"] = {
        str(t): {
            "kept_share": float((m & (sc >= t)).sum() / m.sum()),
            "ok_trusted": float(y[m & (sc >= t)].mean()),
            "ok_flagged": float(y[m & (sc < t)].mean()),
        }
        for t in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    kept = float((m & (sc >= 0.7)).sum() / m.sum())
    q = np.quantile(pred[m], kept)
    out["payoff_smallest_prediction"] = {
        "kept_share": kept,
        "ok_kept": float(y[m & (pred <= q)].mean()),
        "ok_dropped": float(y[m & (pred > q)].mean()),
    }

    # Where the catastrophic misses are, and the ladder without layer-0 edits.
    cat = df[df[f"{level}_catastrophic"]]
    out["catastrophic_location"] = {
        "n": len(cat),
        "layer0": int((cat.layer == 0).sum()),
        "layers0_2": int((cat.layer <= 2).sum()),
        "ablate": int((cat.kind == "ablate").sum()),
        "amplify": int((cat.kind == "amplify").sum()),
    }
    rest = df[df.layer > 0]
    levels = [lv for lv in (*TWIN_LEVELS, *FIX_LEVELS) if f"{lv}_dlogp" in df]
    out["without_layer0"] = {lv: level_metrics(rest, lv) for lv in levels}

    # Error decomposition under different clips (the shares are clip-dependent).
    decomp = {}
    for clip in (1.0, 2.0, 5.0, 20.0):
        e = {lv: df[f"{lv}_err"].clip(upper=clip).mean() for lv in TWIN_LEVELS}
        decomp[str(clip)] = {
            "frozen_layernorm": float((e["replacement"] - e["replacement_ln"]) / e["replacement"]),
            "frozen_attention": float(
                (e["replacement_ln"] - e["replacement_full"]) / e["replacement"]
            ),
        }
    out["decomposition_by_clip"] = decomp
    return out


# ----------------------------------------------------------------------- main
def compute_summary(df: pd.DataFrame) -> tuple[dict, TrustModel]:
    """Every number analyze.py contributes to summary.json, plus the fitted trust model.

    ``df`` must have gone through :func:`add_derived`. One generator (seed ``SEED``) is shared,
    in this order, by the ladder bootstraps, the two trust CV fold shuffles and the fix
    bootstraps: reordering these calls changes the intervals.
    """
    rng = np.random.default_rng(SEED)

    def boot(lv: str) -> dict[str, list[float]]:
        return cluster_bootstrap(df, lambda d: level_metrics(d, lv), BOOT, rng)

    ladder = {lv: boot(lv) for lv in TWIN_LEVELS}

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

    trust, model = trust_analysis(df, TRUST_LEVEL, rng)
    trust_full, _ = trust_analysis(df, "replacement_full", rng)
    cv_scores = trust.pop("_cv_scores")
    trust_full.pop("_cv_scores")
    trust["transfer_lofo"] = trust_transfer(df, TRUST_LEVEL)
    trust["payoff"] = trusted_edges_payoff(df, cv_scores, TRUST_LEVEL)
    robust = robustness(df, cv_scores, TRUST_LEVEL)

    fix_levels = [lv for lv in FIX_COMPARE if f"{lv}_dlogp" in df]
    fix = {
        "levels": fix_levels,
        "ladder": {lv: boot(lv) for lv in fix_levels},
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
        "n_prompts": df.prompt_id.nunique(),
        "levels": list(TWIN_LEVELS),
        "level_labels": LEVEL_LABELS,
        "ladder": ladder,
        "decay": decay,
        "decomposition": decomposition,
        "breakdown": by,
        "trust": trust,
        "trust_replacement_full": {k: trust_full[k] for k in ("cv_auroc", "base_rate")},
        "fix": fix,
        "robustness": robust,
        "real_effect": {
            "median_abs_dlogp": float(df.real_dlogp.abs().median()),
            "frac_big": float(df.real_big.mean()),
            "frac_flip": float(df.real_flip.mean()),
        },
    }
    return summary, model


META = {
    "ladder": {"seed": SEED, "n_boot": BOOT, "unit": "prompt"},
    "trust": {"seed": SEED, "cv_folds": CV_FOLDS, "note": "fold shuffle shares the ladder RNG"},
    "fix": {"seed": SEED, "n_boot": BOOT, "unit": "prompt"},
    "robustness": {"seed": ROBUST_SEED, "n_boot": ROBUST_BOOT, "unit": "prompt"},
}


def print_headline(summary: dict) -> None:
    for k in ("spearman", "within_tol_big", "catastrophic"):
        print(f"ladder {k:15s}", {lv: round(m[k][0], 3) for lv, m in summary["ladder"].items()})
    print("decomposition", {k: round(v, 3) for k, v in summary["decomposition"].items()})
    t = summary["trust"]
    print("trust", {k: round(t[k], 3) for k in ("cv_auroc", "cv_auroc_big", "base_rate")})
    fix = summary["fix"]["ladder"]
    print("fix spearman", {lv: round(m["spearman"][0], 3) for lv, m in fix.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(RESULTS / ("fidelity_small.parquet" if args.small else "fidelity.parquet"))
    df = add_derived(df, [*TWIN_LEVELS, *(f for f in FIX_LEVELS if f"{f}_dlogp" in df)])
    print(f"{len(df)} edits over {df.prompt_id.nunique()} prompts")

    summary, model = compute_summary(df)
    model.save(RESULTS / "trust_model.json")
    model.save(ROOT / "twinlens" / "fidelity" / "trust_model.json")  # shipped with the package
    update_summary(summary, meta=META)  # keeps the generalization / atp_baseline sections
    print_headline(summary)

    from figures import make_figures  # local module, imported late so analysis runs headless

    make_figures(summary, df, FIGS)


if __name__ == "__main__":
    main()
