"""Unit tests for interventions, attribution and patching invariants."""

import torch
from pydantic import TypeAdapter

from twinlens.causal.attribution import direct_logit_attribution, logit_lens
from twinlens.causal.interventions import (
    HeadAblation,
    Intervention,
    NeuronEdit,
    Steering,
    build_hooks,
)
from twinlens.causal.patching import logit_diff_metric, patch_heads, patch_resid
from twinlens.circuits.datasets import ioi_batch


def test_head_ablation_scale_one_is_noop(model):
    toks = model.to_tokens("The cat sat on the mat and then")
    base = model(toks)
    hooks = build_hooks(model, [HeadAblation(layer=3, head=2, scale=1.0)])
    with model.hooks(hooks):
        torch.testing.assert_close(model(toks), base)


def test_head_ablation_only_touches_that_head(model):
    toks = model.to_tokens("The cat sat on the mat and then")
    name = "blocks.4.attn.hook_z"
    _, clean = model.run_with_cache(toks, [name])
    hooks = build_hooks(model, [HeadAblation(layer=4, head=7)])
    seen = {}
    with model.hooks(hooks + [(name, lambda x, n: seen.setdefault("z", x))]):
        model(toks)
    assert torch.all(seen["z"][:, :, 7] == 0)
    torch.testing.assert_close(seen["z"][:, :, :7], clean[name][:, :, :7])


def test_neuron_set_value(model):
    toks = model.to_tokens("Hello there")
    name = "blocks.2.mlp.hook_post"
    seen = {}
    hooks = build_hooks(model, [NeuronEdit(layer=2, neuron=100, value=5.0, positions=[1])])
    with model.hooks(hooks + [(name, lambda x, n: seen.setdefault("p", x))]):
        model(toks)
    assert seen["p"][0, 1, 100] == 5.0


def test_interventions_roundtrip_json():
    adapter = TypeAdapter(list[Intervention])
    specs = adapter.validate_python(
        [
            {"kind": "head", "layer": 9, "head": 9},
            {"kind": "neuron", "layer": 1, "neuron": 3, "scale": 2},
        ]
    )
    assert isinstance(specs[0], HeadAblation) and isinstance(specs[1], NeuronEdit)
    assert adapter.validate_json(adapter.dump_json(specs)) == specs


def test_steering_changes_continuation(model):
    toks = model.to_tokens("I think that you are")
    base = model(toks, last_only=True)
    with model.hooks(build_hooks(model, [Steering(layer=6, plus="Love", minus="Hate", coeff=5)])):
        steered = model(toks, last_only=True)
    assert (steered - base).abs().max() > 0.5


def test_dla_sums_to_logit_difference(model):
    batch = ioi_batch(model, n=4)
    direction = model.W_E[batch.io] - model.W_E[batch.s]
    _, contrib = direct_logit_attribution(model, batch.tokens, direction)
    bias = (direction @ model.ln_f[1]).mean()
    last = model(batch.tokens, last_only=True)[:, -1]
    idx = torch.arange(4)
    actual = (last[idx, batch.io] - last[idx, batch.s]).mean()
    torch.testing.assert_close(contrib.sum() + bias, actual, atol=1e-3, rtol=1e-4)


def test_patching_invariants(model):
    batch = ioi_batch(model, n=2)
    metric = logit_diff_metric(batch.io, batch.s)
    grid = patch_heads(model, batch.tokens, batch.corrupt_tokens, metric)
    assert grid.shape == (12, 12) and torch.isfinite(grid).all()
    # The final residual stream at the last position fully determines the output -> recovery 1.
    resid = patch_resid(model, batch.tokens, batch.corrupt_tokens, metric, hook="hook_resid_post")
    assert abs(resid[-1, -1].item() - 1.0) < 1e-3
    # BOS is identical in the clean and corrupt runs, so patching it changes nothing.
    assert abs(resid[:, 0]).max() < 0.05


def test_logit_lens_last_layer_matches_model(model):
    toks = model.to_tokens("The capital of Japan is the city of")
    lens = logit_lens(model, toks)
    assert lens["final_rank"][-1].eq(0).all()
    assert lens["top_ids"].shape[0] == model.cfg.n_layers + 1
