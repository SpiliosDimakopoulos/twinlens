"""Gemma-2-2B: model parity and twin invariants on the real weights (server only).

Run with ``pytest -m gemma`` after ``python experiments/gemma_prepare.py``; skipped by
default/CI. Needs ~10 GB for the F32 model, ~8 GB for the transcoders and ~12 GB of RAM for the
HF reference. The same invariants run on a tiny random model in ``test_gemma_tiny.py``.
"""

import pytest
import torch

pytestmark = pytest.mark.gemma

# looser than GPT-2: 26 layers of float32 hooks, and Gemma's residual stream is large
CLEAN_TOL = {"atol": 5e-3, "rtol": 1e-4}
AFFINE_TOL = {"atol": 5e-3, "rtol": 1e-3}
PROMPT = "Michael Jordan plays the sport of"


@pytest.fixture(scope="module")
def gemma():
    from twinlens.core.gemma2 import Gemma2

    return Gemma2.load()


@pytest.fixture(scope="module")
def twin(gemma):
    from twinlens.twin.replacement import Twin
    from twinlens.twin.transcoders import GemmaScopeTranscoderSet

    # The hooks the transcoders read/write; set by the convention check (gemma_convention.py).
    tcs = GemmaScopeTranscoderSet(
        in_hook=os.environ.get("TWINLENS_GEMMA_TC_IN", "hook_mlp_in"),
        out_hook=os.environ.get("TWINLENS_GEMMA_TC_OUT", "hook_mlp_out"),
    )
    return Twin(gemma, tcs)


@pytest.fixture(scope="module")
def uncapped_trace(twin):
    """Clean trace without the final logit softcap (the twin is affine before it)."""
    twin.model.final_softcap = None
    try:
        return twin.trace(twin.model.to_tokens(PROMPT))
    finally:
        twin.model.final_softcap = twin.model.cfg.final_softcap


@pytest.fixture(scope="module")
def trace(twin):
    return twin.trace(twin.model.to_tokens(PROMPT))


def _strong(trace, layer):
    pos = trace.n_pos - 1
    f = int(trace.acts[layer][pos].argmax())
    return pos, f, trace.acts[layer][pos, f].item()


@pytest.mark.slow
def test_matches_huggingface(gemma):
    from transformers import AutoModelForCausalLM

    from twinlens.core.gemma2 import REPO
    from twinlens.core.weights import CACHE_DIR

    hf = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.float32, attn_implementation="eager", cache_dir=CACHE_DIR / "hub"
    ).eval()
    toks = gemma.to_tokens("The capital of France is the city of")
    with torch.no_grad():
        ref = hf(toks).logits
    assert (gemma(toks) - ref).abs().max() < 1e-3


def test_bos_is_prepended(gemma):
    assert gemma.to_tokens("Hello")[0, 0].item() == 2


@pytest.mark.parametrize("mode", ["direct", "linear", "linear_lnjac", "linear_fulljac", "real"])
def test_levels_reproduce_clean_run(twin, trace, mode):
    out = twin.run(trace, mode, [None])
    torch.testing.assert_close(out[0], trace.logits[-1], **CLEAN_TOL)


@pytest.mark.parametrize(
    "mode", ["direct", "linear", "linear_fulljac", "replacement", "replacement_full", "real"]
)
def test_noop_edit_reproduces_clean_run(twin, trace, mode):
    from twinlens.twin.replacement import FeatureEdit

    pos, f, a = _strong(trace, 6)
    out = twin.run(trace, mode, [FeatureEdit(6, pos, f, a)])
    torch.testing.assert_close(out[0], trace.logits[-1], **CLEAN_TOL)


@pytest.mark.parametrize("mode", ["linear", "linear_lnjac", "linear_fulljac"])
def test_linear_levels_are_affine(twin, uncapped_trace, mode):
    from twinlens.twin.replacement import FeatureEdit

    trace = uncapped_trace
    pos, f, a = _strong(trace, 12)
    clean = trace.logits[-1]
    twin.model.final_softcap = None
    try:
        full, half = twin.run(
            trace, mode, [FeatureEdit(12, pos, f, 0.0), FeatureEdit(12, pos, f, a / 2)]
        )
    finally:
        twin.model.final_softcap = twin.model.cfg.final_softcap
    torch.testing.assert_close(full - clean, 2 * (half - clean), **AFFINE_TOL)


def test_fulljac_is_first_order_expansion_of_live_norms_and_attention(twin, uncapped_trace):
    """Four RMSNorms per block and softcapped attention: agree with the live twin to first order."""
    from twinlens.twin.replacement import FeatureEdit

    trace = uncapped_trace
    pos, f, a = _strong(trace, 10)
    e = [FeatureEdit(10, pos, f, a * 0.99)]
    clean = trace.logits[-1]
    twin.model.final_softcap = None
    try:
        live = twin.run(trace, "linear_live", e)[0] - clean
        jac = twin.run(trace, "linear_fulljac", e)[0] - clean
        frozen = twin.run(trace, "linear", e)[0] - clean
    finally:
        twin.model.final_softcap = twin.model.cfg.final_softcap
    err = (jac - live).norm() / live.norm()
    assert err < 0.05
    assert err < 0.5 * (frozen - live).norm() / live.norm()
