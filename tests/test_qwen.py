"""Qwen3-0.6B: model parity and twin invariants (needs ~7 GB of prepared transcoders).

Run with ``pytest -m qwen`` after ``python experiments/qwen_prepare.py``; skipped by default/CI.
"""

import pytest
import torch

pytestmark = pytest.mark.qwen


@pytest.fixture(scope="module")
def qwen():
    from twinlens.core.qwen3 import Qwen3

    return Qwen3.load()


@pytest.fixture(scope="module")
def twin(qwen):
    from twinlens.twin.replacement import Twin
    from twinlens.twin.transcoders import QwenTranscoderSet

    return Twin(qwen, QwenTranscoderSet(range(10, 28)))


@pytest.fixture(scope="module")
def trace(twin):
    return twin.trace(twin.model.to_tokens("Michael Jordan plays the sport of"))


def _strong(trace, layer):
    pos = trace.n_pos - 1
    f = int(trace.acts[layer][pos].argmax())
    return pos, f, trace.acts[layer][pos, f].item()


def test_matches_huggingface(qwen):
    from transformers import AutoModelForCausalLM

    from twinlens.core.weights import CACHE_DIR

    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32, cache_dir=CACHE_DIR / "hub"
    ).eval()
    toks = qwen.to_tokens("The capital of France is the city of")
    with torch.no_grad():
        ref = hf(toks).logits
    assert (qwen(toks) - ref).abs().max() < 1e-3


@pytest.mark.parametrize("mode", ["direct", "linear", "linear_lnjac", "linear_fulljac", "real"])
def test_levels_reproduce_clean_run(twin, trace, mode):
    out = twin.run(trace, mode, [None])
    torch.testing.assert_close(out[0], trace.logits[-1], atol=5e-3, rtol=1e-4)


@pytest.mark.parametrize("mode", ["linear", "linear_lnjac", "linear_fulljac"])
def test_linear_levels_are_affine(twin, trace, mode):
    from twinlens.twin.replacement import FeatureEdit

    pos, f, a = _strong(trace, 16)
    clean = trace.logits[-1]
    full, half = twin.run(
        trace, mode, [FeatureEdit(16, pos, f, 0.0), FeatureEdit(16, pos, f, a / 2)]
    )
    torch.testing.assert_close(full - clean, 2 * (half - clean), atol=5e-3, rtol=1e-3)


def test_fulljac_is_first_order_expansion_of_live_norms_and_attention(twin, trace):
    """RMSNorm (incl. QK-norm) and attention Jacobians: agree with the live twin to first order."""
    from twinlens.twin.replacement import FeatureEdit

    pos, f, a = _strong(trace, 14)
    e = [FeatureEdit(14, pos, f, a * 0.99)]
    clean = trace.logits[-1]
    live = twin.run(trace, "linear_live", e)[0] - clean
    jac = twin.run(trace, "linear_fulljac", e)[0] - clean
    frozen = twin.run(trace, "linear", e)[0] - clean
    err = (jac - live).norm() / live.norm()
    assert err < 0.05
    assert err < 0.5 * (frozen - live).norm() / live.norm()


def test_attribution_matches_direct_mode(twin, trace):
    from twinlens.twin.replacement import FeatureEdit

    token = int(trace.logits[-1].argmax())
    for r in twin.attribute_logit(trace, token, top=3):
        out = twin.run(trace, "direct", [FeatureEdit(r["layer"], r["pos"], r["feature"], 0.0)])
        delta = (out[0, token] - trace.logits[-1, token]).item()
        assert delta == pytest.approx(-r["effect"], rel=2e-2, abs=2e-3)
