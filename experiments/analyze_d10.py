"""Analysis of D10 (pre-registered in DECISIONS.md D10 before any number).

    python experiments/analyze_d10.py [--tag _pilot]

Reads ``results/raw_d10_{clt,plt}<tag>/`` (circuits_sweep.py) and the attribution-patching files
of D10-C, and adds ``d10<tag>`` to results/summary.json. Intervals: 95% bootstrap over prompts,
seed 0, 1000 resamples.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from analyze_reanalysis import arrays, boot, load, output_metrics, prompt_groups, rho
from common import RESULTS, update_summary

from twinlens.fidelity.metrics import BIG, within_tol

SEED, BOOT = 0, 1000
MIN_SPAN = 0.5  # prompts with |m(all) - m(none)| below this are excluded (D10-A)
POSITIVE_MEAN = 0.03
PAIRS = [("R3", "R1"), ("R2", "R1"), ("R3", "R4"), ("R3", "R5")]


def log_area(ks: np.ndarray, f: np.ndarray) -> float:
    x = np.log10(ks.astype(float))
    return float(np.trapezoid(f, x) / (x[-1] - x[0])) if len(x) > 1 else float(f[0])


def areas(curves: pd.DataFrame, key: str = "m") -> tuple[pd.DataFrame, int]:
    """Per (pid, truth, ranking): keep- and remove-area; R6 seeds averaged into R6."""
    c = curves[~curves.ranking.str.startswith("prune_")].copy()
    span = c[f"full_{key}"] - c[f"none_{key}"]
    bad = c.loc[span.abs() < MIN_SPAN, "pid"].unique()
    c = c[~c.pid.isin(bad)]
    span = c[f"full_{key}"] - c[f"none_{key}"]
    c["F_keep"] = (c[f"keep_{key}"] - c[f"none_{key}"]) / span
    c["F_remove"] = (c[f"full_{key}"] - c[f"remove_{key}"]) / span
    rows = []
    for (pid, truth, rk), g in c.groupby(["pid", "truth", "ranking"]):
        g = g.sort_values("k")
        ks = g.k.to_numpy()
        rows.append(
            {
                "pid": pid,
                "truth": truth,
                "ranking": rk.split("_")[0],
                "keep": log_area(ks, g.F_keep.to_numpy()),
                "remove": log_area(ks, g.F_remove.to_numpy()),
            }
        )
    a = pd.DataFrame(rows).groupby(["pid", "truth", "ranking"], as_index=False).mean()
    return a, len(bad)


def paired_block(a: pd.DataFrame, truth: str, rng) -> dict:
    w = a[a.truth == truth].pivot(index="pid", columns="ranking", values=["keep", "remove"])
    pids = list(w.index)
    vals = {f"{m}/{r}": w[(m, r)].to_numpy() for m in ("keep", "remove") for r in w["keep"].columns}

    def fn(idx):
        out = {k: float(np.nanmean(v[idx])) for k, v in vals.items()}
        for x, y in PAIRS:
            for m in ("keep", "remove"):
                if f"{m}/{x}" in vals and f"{m}/{y}" in vals:
                    out[f"{m}/{x}-{y}"] = out[f"{m}/{x}"] - out[f"{m}/{y}"]
        return out

    return boot([np.array([i]) for i in range(len(pids))], fn, rng, BOOT) | {"n_prompts": len(pids)}


def verdict(block: dict) -> str:
    mean, lo, hi, _ = block["keep/R3-R1"]
    if lo > 0 and mean >= POSITIVE_MEAN:
        return "positive"
    if hi < 0:
        return "negative"
    return "null"


def prune_block(curves: pd.DataFrame, graphs: pd.DataFrame) -> dict:
    p = curves[curves.ranking.str.startswith("prune_")].copy()
    span = p.full_m - p.none_m
    p = p[span.abs() >= MIN_SPAN]
    p["F_keep"] = (p.keep_m - p.none_m) / (p.full_m - p.none_m)
    out = {}
    for (truth, rk), g in p.groupby(["truth", "ranking"]):
        out[f"{truth}/{rk}/F_keep_mean"] = float(g.F_keep.mean())
    for v, g in graphs.groupby("variant"):
        out[f"{v}/pruned_features_median"] = float(g.n_pruned_features.median())
        out[f"{v}/pruned_edges_median"] = float(g.n_pruned_edges.median())
        out[f"{v}/seconds_median"] = float(g.seconds.median())
    return out


def dose_block(d: pd.DataFrame, rng) -> dict:
    out = {}
    preds = [c for c in d.columns if c.startswith("pred_")]
    for (truth, s), g in d.groupby(["truth", "s"]):
        groups = prompt_groups(g.pid.to_numpy())
        t = g.real_dm.to_numpy()
        P = {c: g[c].to_numpy() for c in preds}

        def fn(idx, t=t, P=P):
            big = np.abs(t[idx]) > BIG
            r = {}
            for c, p in P.items():
                r[f"{c}/spearman"] = rho(p[idx], t[idx])
                r[f"{c}/sign_agree"] = (
                    float(np.mean((p[idx][big] > 0) == (t[idx][big] > 0))) if big.any() else np.nan
                )
                r[f"{c}/within_tol_big"] = (
                    float(np.mean(within_tol(np.abs(p[idx] - t[idx])[big], t[idx][big])))
                    if big.any()
                    else np.nan
                )
            return r

        out[f"{truth}/s={s:g}"] = boot(groups, fn, rng, BOOT) | {"n": len(g)}
    # departure from linearity: real effect at s against (s - 1) times the effect at s = 0 (Δa = -a)
    base = d[d.s == 0.0].set_index(["pid", "truth", "layer", "pos", "feature"]).real_dm
    for s in sorted(set(d.s) - {0.0}):
        g = d[d.s == s].set_index(["pid", "truth", "layer", "pos", "feature"])
        lin = -(s - 1.0) * base.reindex(g.index)
        out[f"departure_median_abs/s={s:g}"] = float((g.real_dm - lin).abs().median())
    return out


def d10c(rng) -> dict:
    out = {}
    specs = {
        "gpt2": ("", "gpt2"),
        "gemma2": ("_gemma2", "gemma2"),
        "gemma2_clt": ("_gemma2_clt", "gemma2_clt"),
    }
    for name, (tag, model) in specs.items():
        variants = {"atp": f"fidelity_atp{tag}.parquet"} | {
            f"atp_{v}": f"fidelity_atp{tag}_detach-{v}.parquet" for v in ("ln", "attn", "both")
        }
        if not all((RESULTS / f).exists() for f in variants.values()):
            continue
        df, _ = load(model)
        for col, f in variants.items():
            df[f"{col}_dlogp"] = pd.read_parquet(RESULTS / f).atp_dlogp.to_numpy()
        lv = ["linear", "linear_fulljac", *variants]
        c = arrays(df, lv)

        def fn(idx, c=c, lv=lv):
            m = {l: output_metrics(c, l, idx) for l in lv}
            r = {f"{l}/{k}": v for l in lv for k, v in m[l].items()}
            r["fix_minus_atp_both/spearman"] = (
                m["linear_fulljac"]["spearman"] - m["atp_both"]["spearman"]
            )
            r["atp_both_minus_linear/spearman"] = (
                m["atp_both"]["spearman"] - m["linear"]["spearman"]
            )
            r["atp_minus_fix/spearman"] = m["atp"]["spearman"] - m["linear_fulljac"]["spearman"]
            return r

        out[name] = boot(prompt_groups(df.prompt_id.to_numpy()), fn, rng, BOOT)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res: dict = {}
    for tc in ("clt", "plt"):
        raw = RESULTS / f"raw_d10_{tc}{args.tag}"
        if not raw.exists():
            continue
        cur = pd.concat([pd.read_parquet(f) for f in sorted(raw.glob("*_curves.parquet"))])
        dose = pd.concat([pd.read_parquet(f) for f in sorted(raw.glob("*_dose.parquet"))])
        gr = pd.concat([pd.read_parquet(f) for f in sorted(raw.glob("*_graphs.parquet"))])
        block: dict = {
            "n_prompts": int(cur.pid.nunique()),
            "top1_ok_share": float(gr.groupby("pid").top1_ok.first().mean()),
        }
        for key in ("m", "dld"):
            a, n_bad = areas(cur, key)
            block[f"areas_{key}"] = {t: paired_block(a, t, rng) for t in ("T1", "T2")}
            block[f"excluded_{key}"] = n_bad
        block["verdict_primary"] = verdict(block["areas_m"]["T1"]) if tc == "clt" else None
        block["prune"] = prune_block(cur, gr)
        block["dose"] = dose_block(dose, rng)
        res[tc] = block
    res["d10c"] = d10c(rng)
    key = f"d10{args.tag}"
    update_summary(
        {key: res}, meta={key: {"seed": SEED, "n_boot": BOOT, "unit": "prompt", "decisions": "D10"}}
    )
    if "clt" in res:
        print(
            "primary (CLT, T1, keep-area R3 - R1):",
            res["clt"]["areas_m"]["T1"]["keep/R3-R1"],
            "->",
            res["clt"]["verdict_primary"],
        )
    for k, v in res["d10c"].items():
        print(k, {q: [round(x, 3) for x in v[q][:3]] for q in v if q.endswith("/spearman")})


if __name__ == "__main__":
    main()
