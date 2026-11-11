"""Model-free tests of the shared fidelity metrics (twinlens.fidelity.metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twinlens.fidelity.metrics import (
    BIG,
    CATASTROPHIC,
    CLIP,
    METRICS,
    TOL_ABS,
    TOL_REL,
    add_derived,
    cluster_bootstrap,
    level_metrics,
    within_tol,
)


def tiny() -> pd.DataFrame:
    """Four prompts x two edits, hand-made so every metric is easy to check by eye."""
    real = [0.0, 0.5, -1.0, 0.02, 2.0, -0.3, 0.04, 1.0]
    pred = [0.05, 0.6, 1.0, 0.0, 9.0, -0.2, 0.2, 0.7]
    return pd.DataFrame(
        {
            "prompt_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "real_dlogp": real,
            "tw_dlogp": pred,
            "tw_cos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


def test_constants():
    assert (TOL_ABS, TOL_REL, CATASTROPHIC, BIG, CLIP) == (0.1, 0.25, 1.0, 0.05, 5.0)


def test_within_tol():
    assert within_tol(0.1, 0.0)
    assert not within_tol(0.11, 0.0)
    assert within_tol(0.35, -1.0)  # 0.1 + 0.25 * |-1|
    assert not within_tol(0.36, 1.0)
    # Predicting "no change" has error |real|: within tolerance iff |real| <= 0.1 / 0.75.
    real = np.array([0.13, 0.14])
    assert within_tol(err=real, real=real).tolist() == [True, False]


def test_add_derived():
    df = add_derived(tiny(), ["tw"])
    assert df.tw_err.tolist() == pytest.approx([0.05, 0.1, 2.0, 0.02, 7.0, 0.1, 0.16, 0.3])
    assert df.tw_ok.tolist() == [True, True, False, True, False, True, False, True]
    assert df.tw_catastrophic.tolist() == [False, False, True, False, True, False, False, False]
    assert df.real_big.tolist() == [False, True, True, False, True, True, False, True]


def test_level_metrics():
    df = add_derived(tiny(), ["tw"])
    m = level_metrics(df, "tw")
    assert list(m) == list(METRICS)
    assert m["median_cos"] == pytest.approx(0.45)
    assert m["within_tol"] == pytest.approx(5 / 8)
    assert m["catastrophic"] == pytest.approx(2 / 8)
    # Big edits: real 0.5, -1.0, 2.0, -0.3, 1.0 vs pred 0.6, 1.0, 9.0, -0.2, 0.7.
    assert m["sign_agree"] == pytest.approx(4 / 5)
    assert m["within_tol_big"] == pytest.approx(3 / 5)
    # The 7.0 error is clipped to CLIP before averaging.
    clipped = [0.05, 0.1, 2.0, 0.02, CLIP, 0.1, 0.16, 0.3]
    assert m["mean_err_clipped"] == pytest.approx(sum(clipped) / 8)
    assert m["spearman"] == pytest.approx(df.real_dlogp.corr(df.tw_dlogp, method="spearman"))
    sub = level_metrics(df, "tw", keys=["catastrophic", "spearman"])
    assert list(sub) == ["catastrophic", "spearman"]
    assert sub == {k: m[k] for k in sub}


def test_cluster_bootstrap_deterministic():
    df = add_derived(tiny(), ["tw"])

    def fn(d):
        return level_metrics(d, "tw", keys=["within_tol", "mean_err_clipped"])

    a = cluster_bootstrap(df, fn, 50, np.random.default_rng(7))
    b = cluster_bootstrap(df, fn, 50, np.random.default_rng(7))
    assert a == b
    assert all(lo <= hi for _, lo, hi in a.values())
    assert a["within_tol"][0] == pytest.approx(5 / 8)


def test_cluster_bootstrap_consumes_generator_in_order():
    """Two calls on one generator give the same as one call per fresh copy in sequence:
    callers rely on this to keep intervals stable when sharing a generator."""
    df = add_derived(tiny(), ["tw"])

    def fn(d):
        return {"n": float(len(d)), "ok": d.tw_ok.mean()}

    rng = np.random.default_rng(0)
    first, second = cluster_bootstrap(df, fn, 20, rng), cluster_bootstrap(df, fn, 20, rng)
    ref = np.random.default_rng(0)
    assert cluster_bootstrap(df, fn, 20, ref) == first
    assert cluster_bootstrap(df, fn, 20, ref) == second
