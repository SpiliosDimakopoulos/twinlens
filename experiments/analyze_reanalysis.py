"""Re-analysis of the existing sweeps (plan v2 phase 0; pre-registered in DECISIONS.md D10-0).

    python experiments/analyze_reanalysis.py [--quick]

No new runs: reads the committed ``fidelity*.parquet``, ``fidelity_atp*.parquet`` and
``edges_*.parquet`` and adds ``reanalysis`` to results/summary.json. Every interval is a 95%
cluster bootstrap over prompts (seed 0, 1000 resamples; ``--quick`` uses 100 and writes nothing).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd
from common import RESULTS, check_edit_keys, update_summary
from scipy.stats import rankdata

from twinlens.fidelity.metrics import BIG, CATASTROPHIC, CLIP, within_tol

SEED = 0
BOOT = 1000
FIX, LN, LIN = "linear_fulljac", "linear_lnjac", "linear"
MODELS = {  # name: (sweep, attribution-patching file or None)
    "gpt2": ("fidelity.parquet", "fidelity_atp.parquet"),
    "qwen3": ("fidelity_qwen3.parquet", None),
    "gemma2": ("fidelity_gemma2.parquet", "fidelity_atp_gemma2.parquet"),
    "gemma2_clt": ("fidelity_gemma2_clt.parquet", "fidelity_atp_gemma2_clt.parquet"),
}
EDGES = {
    "gpt2": "edges_gpt2.parquet",
    "gemma2": "edges_gemma2.parquet",
    "gemma2_clt": "edges_gemma2_clt.parquet",
}
LAYER_BINS = {"0-5": (0, 5), "6-12": (6, 12), "13-19": (13, 19), "20-25": (20, 25)}
TOPK = 10

Arrays = dict[str, np.ndarray]


# ---------------------------------------------------------------- statistics on index arrays
def rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half."""
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(
    groups: list[np.ndarray],
    fn: Callable[[np.ndarray], dict[str, float]],
    rng: np.random.Generator,
    n: int,
) -> dict[str, list[float]]:
    """``{key: [point, lo, hi, p]}``: 95% percentile interval over ``n`` cluster resamples, and the
    two-sided bootstrap p-value of D10-0 (meaningful for differences only)."""
    point = fn(np.concatenate(groups))
    k = len(groups)
    samples = [fn(np.concatenate([groups[i] for i in rng.integers(0, k, k)])) for _ in range(n)]
    out = {}
    for key, v in point.items():
        s = np.array([x[key] for x in samples], dtype=float)
        s = s[~np.isnan(s)]
        p = min(1.0, 2 * min(np.mean(s <= 0), np.mean(s >= 0))) if len(s) else float("nan")
        lo, hi = np.percentile(s, [2.5, 97.5]) if len(s) else (np.nan, np.nan)
        out[key] = [float(v), float(lo), float(hi), float(p)]
    return out


def prompt_groups(pid: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(pid == p) for p in np.unique(pid)]


# ---------------------------------------------------------------- data
def load(name: str) -> tuple[pd.DataFrame, list[str]]:
    sweep, atp = MODELS[name]
    df = pd.read_parquet(RESULTS / sweep)
    levels = [LIN, LN, FIX]
    if atp:
        a = pd.read_parquet(RESULTS / atp)
        check_edit_keys(df, a, f"{atp} vs {sweep}")
        df["atp_dlogp"] = a.atp_dlogp.to_numpy()
        levels.append("atp")
    df["zero_dlogp"] = 0.0
    return df.reset_index(drop=True), levels


def arrays(df: pd.DataFrame, levels: list[str], truth: str = "real") -> Arrays:
    c = {lv: df[f"{lv}_dlogp"].to_numpy(float) for lv in [*levels, "zero"]}
    c["truth"] = df[f"{truth}_dlogp"].to_numpy(float)
    return c


def output_metrics(c: Arrays, lv: str, idx: np.ndarray) -> dict[str, float]:
    """The report's four headline metrics (as ``level_metrics``) on the edits ``idx``."""
    p, t = c[lv][idx], c["truth"][idx]
    big = np.abs(t) > BIG
    err = np.abs(p - t)
    return {
        "spearman": rho(p, t),
        "sign_agree": float(np.mean((p[big] > 0) == (t[big] > 0))) if big.any() else np.nan,
        "within_tol_big": float(np.mean(within_tol(err[big], t[big]))) if big.any() else np.nan,
        "catastrophic": float(np.mean(err > CATASTROPHIC)),
    }


def big_metrics(c: Arrays, lv: str, idx: np.ndarray) -> dict[str, float]:
    """0.3: metrics restricted to big edits."""
    p, t = c[lv][idx], c["truth"][idx]
    b = np.abs(t) > BIG
    p, t = p[b], t[b]
    err = np.abs(p - t)
    out = {
        "within_tol": float(np.mean(within_tol(err, t))) if len(t) else np.nan,
        "mean_err_clipped": float(np.mean(np.minimum(err, CLIP))) if len(t) else np.nan,
        "n": float(len(t)),
    }
    if lv != "zero":
        out["spearman"] = rho(p, t)
        out["sign_agree"] = float(np.mean((p > 0) == (t > 0))) if len(t) else np.nan
    return out


# ---------------------------------------------------------------- the five analyses
def a01_flag(df: pd.DataFrame, levels: list[str], rng, n) -> dict:
    c = arrays(df, levels)
    t = c["truth"]
    wrong = ~within_tol(np.abs(c[LIN] - t), t)
    big = np.abs(t) > BIG
    scores = {
        "dis_full": np.abs(c[LIN] - c[FIX]),
        "dis_ln": np.abs(c[LIN] - c[LN]),
        "size": np.abs(c[LIN]),
    }

    def fn(idx):
        idx = idx[big[idx]]
        y = wrong[idx]
        out = {k: auroc(s[idx], y) for k, s in scores.items()}
        out["dis_full_minus_size"] = out["dis_full"] - out["size"]
        out["wrong_share"] = float(y.mean())
        return out

    res = boot(prompt_groups(df.prompt_id.to_numpy()), fn, rng, n)
    a, lo = res["dis_full"][0], res["dis_full_minus_size"][1]
    res["reading"] = "diagnostic use" if (a >= 0.80 and lo > 0) else "no diagnostic use"
    return res


def a02_ct_output(df: pd.DataFrame, rng, n) -> dict:
    c = arrays(df, [LIN, LN, FIX], truth="real_frozen_attn")

    def fn(idx):
        r = {lv: rho(c[lv][idx], c["truth"][idx]) for lv in (LIN, LN, FIX)}
        return r | {
            "ln_minus_linear": r[LN] - r[LIN],
            "fix_minus_linear": r[FIX] - r[LIN],
            "ln_minus_fix": r[LN] - r[FIX],
        }

    return boot(prompt_groups(df.prompt_id.to_numpy()), fn, rng, n)


def a02_ct_edges(name: str, rng, n) -> dict:
    cols = [
        "prompt_id",
        "linear_dact",
        "linear_lnjac_dact",
        "linear_fulljac_dact",
        "real_frozen_attn_dact",
    ]
    e = pd.read_parquet(RESULTS / EDGES[name], columns=cols)
    c = {lv: e[f"{lv}_dact"].to_numpy(float) for lv in (LIN, LN, FIX)}
    t = e.real_frozen_attn_dact.to_numpy(float)
    groups = prompt_groups(e.prompt_id.to_numpy())
    del e

    def fn(idx):
        rt = rankdata(t[idx])
        r = {lv: float(np.corrcoef(rankdata(c[lv][idx]), rt)[0, 1]) for lv in (LIN, LN, FIX)}
        return r | {
            "ln_minus_linear": r[LN] - r[LIN],
            "fix_minus_linear": r[FIX] - r[LIN],
            "ln_minus_fix": r[LN] - r[FIX],
        }

    return boot(groups, fn, rng, n) | {"n_pairs": float(len(t))}


def a03_where(df: pd.DataFrame, levels: list[str], rng, n) -> dict:
    c = arrays(df, levels)
    lvls = [*levels, "zero"]
    out: dict = {}
    sel = df.selection.to_numpy()
    for name, mask in {
        "all": np.ones(len(df), bool),
        "top": sel == "top",
        "random": sel == "random",
    }.items():
        sub = df[mask]
        rows = np.flatnonzero(mask)
        groups = [rows[g] for g in prompt_groups(sub.prompt_id.to_numpy())]

        def fn(idx):
            return {f"{lv}/{k}": v for lv in lvls for k, v in big_metrics(c, lv, idx).items()}

        out[name] = boot(groups, fn, rng, n)

    # top-10 recall on ablations, one value per prompt, then a bootstrap of the mean
    abl = df[df.kind == "ablate"]
    per_prompt = []
    for _, g in abl.groupby("prompt_id"):
        real_top = set(np.argsort(-np.abs(g.real_dlogp.to_numpy()), kind="stable")[:TOPK])
        row = {"chance": TOPK / len(g)}
        for lv in levels:
            pred_top = np.argsort(-np.abs(g[f"{lv}_dlogp"].to_numpy()), kind="stable")[:TOPK]
            row[lv] = len(real_top.intersection(pred_top)) / TOPK
        per_prompt.append(row)
    pp = pd.DataFrame(per_prompt)
    vals = {k: pp[k].to_numpy() for k in pp}

    def mean_fn(idx):
        m = {k: float(v[idx].mean()) for k, v in vals.items()}
        if "atp" in m:
            m["fix_minus_atp"] = m[FIX] - m["atp"]
        return m | {"fix_minus_linear": m[FIX] - m[LIN]}

    out["top10_recall"] = boot([np.array([i]) for i in range(len(pp))], mean_fn, rng, n)
    return out


def a04_depth(plt: pd.DataFrame, clt: pd.DataFrame, rng, n) -> dict:
    keep = np.intersect1d(plt.prompt_id.unique(), clt.prompt_id.unique())
    both = pd.concat(
        [
            plt[plt.prompt_id.isin(keep)].assign(tc="plt"),
            clt[clt.prompt_id.isin(keep)].assign(tc="clt"),
        ],
        ignore_index=True,
    )
    p = {lv: both[f"{lv}_dlogp"].to_numpy(float) for lv in (LIN, FIX)}
    t = both.real_dlogp.to_numpy(float)
    tc = both.tc.to_numpy()
    layer = both.layer.to_numpy()

    def fn(idx):
        out = {}
        for b, (lo, hi) in LAYER_BINS.items():
            for lv in (LIN, FIX):
                r = {}
                for name in ("plt", "clt"):
                    i = idx[(tc[idx] == name) & (layer[idx] >= lo) & (layer[idx] <= hi)]
                    r[name] = rho(p[lv][i], t[i])
                    out[f"{b}/{lv}/{name}"] = r[name]
                out[f"{b}/{lv}/clt_minus_plt"] = r["clt"] - r["plt"]
        return out

    return boot(prompt_groups(both.prompt_id.to_numpy()), fn, rng, n)


def a05_tests(df: pd.DataFrame, levels: list[str], rng, n) -> dict:
    c = arrays(df, levels)
    pairs = [(FIX, LIN)] + ([(FIX, "atp")] if "atp" in levels else [])

    def fn(idx):
        out = {}
        for a, b in pairs:
            ma, mb = output_metrics(c, a, idx), output_metrics(c, b, idx)
            out |= {f"{a}-{b}/{m}": ma[m] - mb[m] for m in ma}
            out |= {f"{a}/{m}": ma[m] for m in ma} | {f"{b}/{m}": mb[m] for m in mb}
        return out

    return boot(prompt_groups(df.prompt_id.to_numpy()), fn, rng, n)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    order = sorted(pvals, key=pvals.get)
    m, out, still = len(order), {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        still = still and pvals[k] <= thr
        out[k] = {"p": pvals[k], "holm_threshold": thr, "survives": bool(still)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 100 if args.quick else BOOT
    rng = np.random.default_rng(SEED)
    res: dict = {}
    data = {m: load(m) for m in MODELS}

    res["flag"] = {m: a01_flag(df, lv, rng, n) for m, (df, lv) in data.items()}
    print(
        "0.1 done",
        {
            m: (r["dis_full"][:3], r["dis_full_minus_size"][:3], r["reading"])
            for m, r in res["flag"].items()
        },
        flush=True,
    )

    res["ct_truth"] = {"output_gemma2_clt": a02_ct_output(data["gemma2_clt"][0], rng, n)}
    for m in EDGES:
        res["ct_truth"][f"edges_{m}"] = a02_ct_edges(m, rng, n)
        print("0.2 edges", m, flush=True)

    res["where"] = {m: a03_where(df, lv, rng, n) for m, (df, lv) in data.items()}
    print("0.3 done", flush=True)

    res["depth"] = a04_depth(data["gemma2"][0], data["gemma2_clt"][0], rng, n)

    tests = {m: a05_tests(df, lv, rng, n) for m, (df, lv) in data.items()}
    res["tests"] = tests
    pv = {
        f"{m}:{k}": v[3] for m, t in tests.items() for k, v in t.items() if "-" in k.split("/")[0]
    }
    res["holm"] = holm(pv)
    print("0.5", sum(v["survives"] for v in res["holm"].values()), "of", len(pv), "survive Holm")

    if not args.quick:
        update_summary(
            {"reanalysis": res},
            meta={
                "reanalysis": {"seed": SEED, "n_boot": BOOT, "unit": "prompt", "decisions": "D10-0"}
            },
        )


if __name__ == "__main__":
    main()
