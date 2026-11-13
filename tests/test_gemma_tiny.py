"""Gemma-2 port on a tiny random model: HF parity and the twin invariants (no downloads).

The model is a random ``transformers`` Gemma2ForCausalLM with every Gemma-specific part switched
on (four norms with random weights, GQA, sliding-window layers, both softcaps biting), and the
transcoders are random JumpReLU ones saved as Gemma Scope-style ``params.npz`` files.
"""

from contextlib import contextmanager

import numpy as np
import pytest
import torch

from twinlens.twin.replacement import LEVELS, FeatureEdit, Twin

CLEAN_TOL = {"atol": 1e-4, "rtol": 1e-4}
AFFINE_TOL = {"atol": 1e-4, "rtol": 1e-3}
N_LAYERS, D_SAE, N_POS = 3, 256, 10
HOOKS = [
    (i, o) for i in ("hook_mlp_in", "ln2.hook_normalized") for o in ("hook_mlp_out", "mlp.hook_out")
]


@pytest.fixture(scope="module")
def hf_and_path(tmp_path_factory):
    from safetensors.torch import save_file
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    cfg = Gemma2Config(
        num_hidden_layers=N_LAYERS,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=512,
        max_position_embeddings=64,
        query_pre_attn_scalar=16,
        attn_logit_softcapping=5.0,
        final_logit_softcapping=30.0,
        sliding_window=4,
        attn_implementation="eager",
    )
    hf = Gemma2ForCausalLM(cfg).eval()
    with torch.no_grad():  # make the norms, attention softcap and final softcap all matter
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.normal_(0, 0.3)
            elif "q_proj" in name or "k_proj" in name:
                p.mul_(6)
            elif "mlp" in name:  # O(1) MLP outputs, so the pre-post-norm hook is well scaled
                p.mul_(5)
        hf.model.embed_tokens.weight.mul_(30)
    path = tmp_path_factory.mktemp("gemma") / "model.safetensors"
    save_file(
        {k: v.contiguous() for k, v in hf.state_dict().items() if k != "lm_head.weight"}, path
    )
    return hf, path


@pytest.fixture(scope="module")
def model(hf_and_path):
    from twinlens.core.gemma2 import Gemma2, GemmaConfig

    hf, path = hf_and_path
    return Gemma2.from_files(path, GemmaConfig.from_hf(hf.config))


@pytest.fixture(scope="module")
def tc_files(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("tc")
    files = {}
    for layer in range(N_LAYERS):
        files[layer] = d / f"layer_{layer}.npz"
        np.savez(
            files[layer],
            W_enc=rng.normal(0, 0.3, (64, D_SAE)).astype(np.float32),
            W_dec=rng.normal(0, 0.1, (D_SAE, 64)).astype(np.float32),
            b_enc=rng.normal(-0.5, 0.3, D_SAE).astype(np.float32),
            b_dec=rng.normal(0, 0.1, 64).astype(np.float32),
            threshold=rng.uniform(0.05, 0.5, D_SAE).astype(np.float32),
        )
    return files


@pytest.fixture(scope="module")
def tokens():
    g = torch.Generator().manual_seed(1)
    return torch.cat([torch.tensor([2]), torch.randint(3, 512, (N_POS - 1,), generator=g)])[None]


def make_twin(model, tc_files, in_hook="hook_mlp_in", out_hook="hook_mlp_out"):
    from twinlens.twin.transcoders import GemmaScopeTranscoderSet

    tcs = GemmaScopeTranscoderSet(range(N_LAYERS), in_hook, out_hook, files=tc_files)
    return Twin(model, tcs)


@contextmanager
def uncapped(model):
    """Without the final logit softcap the linear levels must be exactly affine."""
    cap, model.final_softcap = model.final_softcap, None
    try:
        yield
    finally:
        model.final_softcap = cap


def _strong(trace, layer):
    pos = trace.n_pos - 1
    f = int(trace.acts[layer][pos].argmax())
    return pos, f, trace.acts[layer][pos, f].item()


def test_matches_huggingface(hf_and_path, model):
    hf, _ = hf_and_path
    toks = torch.randint(0, 512, (2, 12), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = hf(toks).logits
    assert ref.abs().max() > 15  # the final softcap is not a no-op here
    assert (model(toks) - ref).abs().max() < 1e-4


def test_sliding_window_matters(model):
    toks = torch.randint(0, 512, (1, 12), generator=torch.Generator().manual_seed(3))
    full = model(toks)
    window = model.cfg.sliding_window
    model.cfg.__dict__["sliding_window"] = 100  # frozen dataclass: bypass for this check
    try:
        assert (model(toks) - full).abs().max() > 1e-3
    finally:
        model.cfg.__dict__["sliding_window"] = window


def test_jumprelu_threshold(model, tc_files, tokens):
    twin = make_twin(model, tc_files)
    trace = twin.trace(tokens)
    for layer in range(N_LAYERS):
        tc = twin.tcs[layer]
        _, cache = model.run_with_cache(tokens, [f"blocks.{layer}.hook_mlp_in"])
        pre = tc.pre(cache[f"blocks.{layer}.hook_mlp_in"][0])
        on = trace.acts[layer] > 0
        assert torch.equal(on, pre > tc.threshold)
        assert on.any() and ((pre > 0) & ~on).any()  # thresholds switch off some ReLU-positives


@pytest.mark.parametrize("mode", [*LEVELS, "real"])
def test_levels_reproduce_clean_run(model, tc_files, tokens, mode):
    twin = make_twin(model, tc_files)
    trace = twin.trace(tokens)
    out = twin.run(trace, mode, [None])
    torch.testing.assert_close(out[0], trace.logits[-1], **CLEAN_TOL)


@pytest.mark.parametrize("hooks", HOOKS)
@pytest.mark.parametrize("mode", [*LEVELS, "real"])
def test_noop_edit_reproduces_clean_run(model, tc_files, tokens, mode, hooks):
    """Installs every twin hook from layer 0 on: transcoder + error term must be exact,
    for every choice of transcoder input/output hook."""
    twin = make_twin(model, tc_files, *hooks)
    trace = twin.trace(tokens)
    pos, f, a = _strong(trace, 0)
    out = twin.run(trace, mode, [FeatureEdit(0, pos, f, a)])
    torch.testing.assert_close(out[0], trace.logits[-1], **CLEAN_TOL)


@pytest.mark.parametrize("hooks", HOOKS)
@pytest.mark.parametrize("mode", ["direct", "linear", "linear_lnjac", "linear_fulljac"])
def test_linear_levels_are_affine(model, tc_files, tokens, mode, hooks):
    twin = make_twin(model, tc_files, *hooks)
    with uncapped(model):
        trace = twin.trace(tokens)
        pos, f, a = _strong(trace, 0)
        clean = trace.logits[-1]
        full, half = twin.run(
            trace, mode, [FeatureEdit(0, pos, f, 0.0), FeatureEdit(0, pos, f, a / 2)]
        )
    torch.testing.assert_close(full - clean, 2 * (half - clean), **AFFINE_TOL)


def _rel_errors(model, tc_files, tokens, hooks, jac, live, frozen="linear"):
    twin = make_twin(model, tc_files, *hooks)
    with uncapped(model):
        trace = twin.trace(tokens)
        pos, f, a = _strong(trace, 0)
        e = [FeatureEdit(0, pos, f, a * 0.99)]
        clean = trace.logits[-1]
        ref = twin.run(trace, live, e)[0] - clean
        approx = twin.run(trace, jac, e)[0] - clean
        base = twin.run(trace, frozen, e)[0] - clean
    return ((approx - ref).norm() / ref.norm()).item(), ((base - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("hooks", HOOKS)
def test_fulljac_is_first_order_expansion_of_live_norms_and_attention(
    model, tc_files, tokens, hooks
):
    """All four RMSNorms and the softcapped attention: agree with the live twin to first order."""
    err, frozen = _rel_errors(model, tc_files, tokens, hooks, "linear_fulljac", "linear_live")
    assert err < 0.05
    assert err < 0.5 * frozen


@pytest.mark.parametrize("hooks", HOOKS)
def test_lnjac_is_first_order_expansion_of_live_norms(model, tc_files, tokens, hooks):
    err, frozen = _rel_errors(model, tc_files, tokens, hooks, "linear_lnjac", "linear_lnlive")
    assert err < 0.05
    assert err < 0.5 * frozen


def test_attn_softcap_derivative_is_needed(model, tc_files, tokens, monkeypatch):
    """Dropping the softcap derivative from the attention Jacobian visibly breaks agreement."""
    err, _ = _rel_errors(model, tc_files, tokens, HOOKS[0], "linear_fulljac", "linear_live")
    twin = make_twin(model, tc_files)
    build, cap = twin._attn_jacobian_hooks, model.cfg.attn_softcap

    def without_cap(trace):  # the model keeps its softcap; only the Jacobian forgets it
        model.cfg.__dict__["attn_softcap"] = None  # frozen dataclass
        try:
            return build(trace)
        finally:
            model.cfg.__dict__["attn_softcap"] = cap

    monkeypatch.setattr(twin, "_attn_jacobian_hooks", without_cap)
    with uncapped(model):
        trace = twin.trace(tokens)
        pos, f, a = _strong(trace, 0)
        e = [FeatureEdit(0, pos, f, a * 0.99)]
        clean = trace.logits[-1]
        live = twin.run(trace, "linear_live", e)[0] - clean
        nocap = twin.run(trace, "linear_fulljac", e)[0] - clean
    assert ((nocap - live).norm() / live.norm()).item() > 2 * err


def test_attribution_matches_direct_mode(model, tc_files, tokens):
    twin = make_twin(model, tc_files)
    with uncapped(model):
        trace = twin.trace(tokens)
        token = int(trace.logits[-1].argmax())
        for r in twin.attribute_logit(trace, token, top=3):
            edit = FeatureEdit(r["layer"], r["pos"], r["feature"], 0.0)
            delta = (twin.run(trace, "direct", [edit])[0, token] - trace.logits[-1, token]).item()
            assert delta == pytest.approx(-r["effect"], rel=2e-2, abs=2e-3)
