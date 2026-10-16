"""Invariants of the digital twin (transcoder replacement model)."""

import pytest
import torch

from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS, FeatureEdit, Twin


@pytest.fixture(scope="module")
def twin(model):
    return Twin(model)


@pytest.fixture(scope="module")
def trace(twin):
    return twin.trace(twin.model.to_tokens("The Colosseum is located in the city of"))


def _strong_feature(trace, layer):
    pos = trace.n_pos - 2
    f = int(trace.acts[layer][pos].argmax())
    return pos, f, trace.acts[layer][pos, f].item()


@pytest.mark.parametrize("mode", [*TWIN_LEVELS, *FIX_LEVELS, "real"])
def test_all_levels_reproduce_clean_run(twin, trace, mode):
    out = twin.run(trace, mode, [None])
    torch.testing.assert_close(out[0], trace.logits[-1], atol=2e-3, rtol=1e-4)


def test_noop_edit_is_exact(twin, trace):
    pos, f, a = _strong_feature(trace, 6)
    for mode in [*TWIN_LEVELS, *FIX_LEVELS, "real"]:
        out = twin.run(trace, mode, [FeatureEdit(6, pos, f, a)])
        torch.testing.assert_close(out[0], trace.logits[-1], atol=2e-3, rtol=1e-4)


def test_linear_twin_is_affine(twin, trace):
    pos, f, a = _strong_feature(trace, 4)
    clean = trace.logits[-1]
    full, half = twin.run(
        trace, "linear", [FeatureEdit(4, pos, f, 0.0), FeatureEdit(4, pos, f, a / 2)]
    )
    torch.testing.assert_close(full - clean, 2 * (half - clean), atol=2e-3, rtol=1e-3)


def test_batched_rows_are_independent(twin, trace):
    pos, f, _ = _strong_feature(trace, 5)
    e = FeatureEdit(5, pos, f, 0.0)
    both = twin.run(trace, "replacement", [e, None])
    single = twin.run(trace, "replacement", [e])
    torch.testing.assert_close(both[0], single[0], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(both[1], trace.logits[-1], atol=2e-3, rtol=1e-4)


def test_attribution_matches_direct_mode(twin, trace):
    """Gradient x activation (attribution) must equal the 'direct' simulation of an ablation."""
    token = int(trace.logits[-1].argmax())
    rows = twin.attribute_logit(trace, token, top=3)
    assert rows, "expected active features with an effect"
    for r in rows:
        out = twin.run(trace, "direct", [FeatureEdit(r["layer"], r["pos"], r["feature"], 0.0)])
        delta = (out[0, token] - trace.logits[-1, token]).item()
        assert delta == pytest.approx(-r["effect"], rel=1e-2, abs=1e-3)


def test_lnjac_twin_is_affine(twin, trace):
    pos, f, a = _strong_feature(trace, 4)
    clean = trace.logits[-1]
    full, half = twin.run(
        trace, "linear_lnjac", [FeatureEdit(4, pos, f, 0.0), FeatureEdit(4, pos, f, a / 2)]
    )
    torch.testing.assert_close(full - clean, 2 * (half - clean), atol=2e-3, rtol=1e-3)


def test_lnjac_is_the_first_order_expansion_of_live_layernorm(twin, trace):
    """For a tiny edit, keeping LN's derivative must match live LN to first order;
    the frozen-scale twin must not (it drops a first-order term)."""
    pos, f, a = _strong_feature(trace, 3)
    e = [FeatureEdit(3, pos, f, a * 0.99)]
    clean = trace.logits[-1]
    live = twin.run(trace, "linear_lnlive", e)[0] - clean
    jac = twin.run(trace, "linear_lnjac", e)[0] - clean
    frozen = twin.run(trace, "linear", e)[0] - clean
    err_jac = (jac - live).norm() / live.norm()
    err_frozen = (frozen - live).norm() / live.norm()
    assert err_jac < 0.05
    assert err_jac < 0.5 * err_frozen


def test_fulljac_twin_is_affine(twin, trace):
    pos, f, a = _strong_feature(trace, 4)
    clean = trace.logits[-1]
    full, half = twin.run(
        trace, "linear_fulljac", [FeatureEdit(4, pos, f, 0.0), FeatureEdit(4, pos, f, a / 2)]
    )
    torch.testing.assert_close(full - clean, 2 * (half - clean), atol=2e-3, rtol=1e-3)


def test_fulljac_is_the_first_order_expansion_of_live_attention(twin, trace):
    pos, f, a = _strong_feature(trace, 3)
    e = [FeatureEdit(3, pos, f, a * 0.99)]
    clean = trace.logits[-1]
    live = twin.run(trace, "linear_live", e)[0] - clean
    jac = twin.run(trace, "linear_fulljac", e)[0] - clean
    lnjac = twin.run(trace, "linear_lnjac", e)[0] - clean
    err_jac = (jac - live).norm() / live.norm()
    assert err_jac < 0.05
    assert err_jac < (lnjac - live).norm() / live.norm()
