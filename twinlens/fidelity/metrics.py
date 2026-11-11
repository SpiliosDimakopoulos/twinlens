"""Fidelity metrics shared by the analysis scripts (``experiments/analyze*.py``).

A twin's prediction of an edit's effect on log p(target) is *within tolerance* when
``|error| <= TOL_ABS + TOL_REL * |real effect|``, *catastrophic* when ``|error| > CATASTROPHIC``
nats, and an edit has a *big* (noticeable) real effect when ``|real| > BIG``. Uncertainty is a
cluster bootstrap over prompts, since edits within one prompt are not independent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

TOL_ABS = 0.1  # nats
TOL_REL = 0.25  # share of |real effect|
CATASTROPHIC = 1.0  # nats of |error|
BIG = 0.05  # nats of |real effect|: an edit that matters
CLIP = 5.0  # nats; errors are clipped here before averaging so one blow-up can't dominate

METRICS = (
    "spearman",
    "median_cos",
    "sign_agree",
    "spearman_big",
    "within_tol_big",
    "within_tol",
    "catastrophic",
    "mean_err_clipped",
)


def within_tol(err, real):
    """``|err| <= TOL_ABS + TOL_REL * |real|``, elementwise (Series, arrays or scalars)."""
    return err <= TOL_ABS + TOL_REL * abs(real)


def add_derived(df: pd.DataFrame, levels: Iterable[str]) -> pd.DataFrame:
    """Add ``{lv}_err``, ``{lv}_ok``, ``{lv}_catastrophic`` per level, and ``real_big``.

    Modifies ``df`` in place and returns it."""
    for lv in levels:
        err = (df[f"{lv}_dlogp"] - df.real_dlogp).abs()
        df[f"{lv}_err"] = err
        df[f"{lv}_ok"] = within_tol(err, df.real_dlogp)
        df[f"{lv}_catastrophic"] = err > CATASTROPHIC
    df["real_big"] = df.real_dlogp.abs() > BIG
    return df


def _spearman(d: pd.DataFrame, lv: str) -> float:
    return d[[f"{lv}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1]


def level_metrics(
    d: pd.DataFrame, lv: str, keys: Iterable[str] | None = None
) -> dict[str, float]:
    """Fidelity metrics of twin level ``lv`` on the edits ``d`` (needs :func:`add_derived`).

    ``keys`` restricts (and orders) the metrics computed; default: all of :data:`METRICS`.
    """
    big = d[d.real_big]
    fns: dict[str, Callable[[], float]] = {
        "spearman": lambda: _spearman(d, lv),
        "median_cos": lambda: d[f"{lv}_cos"].median(),
        "sign_agree": lambda: ((big[f"{lv}_dlogp"] > 0) == (big.real_dlogp > 0)).mean(),
        "spearman_big": lambda: _spearman(big, lv),
        "within_tol_big": lambda: big[f"{lv}_ok"].mean(),
        "within_tol": lambda: d[f"{lv}_ok"].mean(),
        "catastrophic": lambda: d[f"{lv}_catastrophic"].mean(),
        "mean_err_clipped": lambda: d[f"{lv}_err"].clip(upper=CLIP).mean(),
    }
    return {k: fns[k]() for k in (METRICS if keys is None else keys)}


def cluster_bootstrap(
    df: pd.DataFrame,
    fn: Callable[[pd.DataFrame], Mapping[str, float]],
    n: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """95% cluster bootstrap over prompts: ``{metric: [point, lo, hi]}``.

    Draws ``n`` resamples of whole prompts (with replacement) from ``rng``, in order, so a fixed
    generator state gives identical intervals. NaN resamples are ignored in the percentiles.
    """
    point = fn(df)
    groups = dict(iter(df.groupby("prompt_id")))
    ids = list(groups)
    samples = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        samples.append(fn(pd.concat([groups[i] for i in pick], ignore_index=True)))
    out = {}
    for k, v in point.items():
        vals = np.array([s[k] for s in samples], dtype=float)
        out[k] = [float(v), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return out
