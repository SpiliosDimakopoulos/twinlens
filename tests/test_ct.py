"""Tests for twinlens.ct on a tiny circuit-tracer model (skipped without circuit-tracer).

The key check: the graph's path sums (``total_effects``) equal a finite difference in the exact
linear model the graph describes, i.e. the replacement model with frozen gates and error terms,
and with attention patterns and normalisation scales either frozen (today's graph) or live
(the graph of circuit-tracer PR #123 with both flags off, to first order).
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ct = pytest.importorskip("circuit_tracer")
nn = torch.nn

from twinlens.ct import graph_math as gm  # noqa: E402
from twinlens.ct import interventions as iv  # noqa: E402

PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
TARGET = 5
HAS_FLAGS = "freeze_attention" in inspect.signature(ct.attribute).parameters


def tiny_llama():
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from circuit_tracer.transcoder.activation_functions import TopK
    from transformer_lens import HookedTransformerConfig

    cfg = HookedTransformerConfig.unwrap(
        {
            "n_layers": 3,
            "d_model": 8,
            "n_ctx": 64,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "silu",
            "d_vocab": 16,
            "d_vocab_out": 16,
            "eps": 1e-5,
            "model_name": "Llama-3.2-1B",
            "original_architecture": "LlamaForCausalLM",
            "tokenizer_name": "gpt2",
            "normalization_type": "RMSPre",
            "final_rms": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": 4,
            "rotary_base": 500000.0,
            "gated_mlp": True,
            "n_key_value_heads": 2,
            "use_attn_scale": True,
            "attn_scale": np.float64(2.0),
            "init_weights": False,
            "dtype": torch.float32,
            "device": "cpu",
            "default_prepend_bos": True,
        }
    )
    torch.manual_seed(0)
    tcs = {
        layer: SingleLayerTranscoder(
            cfg.d_model, cfg.d_model * 4, TopK(6), layer, skip_connection=False
        )
        for layer in range(cfg.n_layers)
    }
    for tc in tcs.values():
        for _, p in tc.named_parameters():
            nn.init.uniform_(p, a=-1, b=1)
    model = ReplacementModel.from_config(
        cfg,
        TranscoderSet(tcs, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out"),
    )
    for _, p in model.named_parameters():
        nn.init.uniform_(p, a=-1, b=1)
    return model


@contextmanager
def special_ids(model, ids):
    cls = type(model.tokenizer)
    old = cls.all_special_ids
    try:
        cls.all_special_ids = property(lambda self: ids)
        yield
    finally:
        cls.all_special_ids = old


@pytest.fixture(scope="module")
def model():
    return tiny_llama()


def graph_of(model, **flags):
    with special_ids(model, [0]):
        return ct.attribute(
            PROMPT, model, attribution_targets=torch.tensor([TARGET]), batch_size=64, **flags
        )


def twin_logit(model, edit, eps, live_attention: bool, live_norm: bool) -> float:
    """Demeaned target logit of the replacement model with frozen gates and error terms, after
    scaling feature ``edit = (layer, pos, feat)`` by (1 + eps)."""
    names = lambda n: "hook_pattern" in n or "hook_scale" in n or "mlp.hook_" in n  # noqa: E731
    with torch.no_grad():
        _, cache = model.run_with_cache(PROMPT, names_filter=names)
    layer_e, pos_e, feat_e = edit
    inputs: dict[int, torch.Tensor] = {}

    def freeze(out, hook):
        return cache[hook.name]

    def record_in(x, hook):
        inputs[hook.layer()] = x
        return x

    def mlp_out(out, hook):
        layer = hook.layer()
        tc = model.transcoders[layer]
        x0 = cache[f"blocks.{layer}.mlp.hook_in"]
        pre0 = x0 @ tc.W_enc.T + tc.b_enc
        active = tc.activation_function(pre0) != 0
        active[:, 0] = False  # circuit-tracer drops features at the special (BOS) position
        dpre = (inputs[layer] - x0) @ tc.W_enc.T
        dact = torch.where(active, dpre, torch.zeros_like(dpre))
        if layer == layer_e:
            dact[0, pos_e, feat_e] += eps * tc.activation_function(pre0)[0, pos_e, feat_e]
        return cache[hook.name] + dact @ tc.W_dec

    hooks = [(n, record_in) for n in cache if n.endswith("mlp.hook_in")]
    hooks += [(n, mlp_out) for n in cache if n.endswith("mlp.hook_out")]
    if not live_attention:
        hooks += [(n, freeze) for n in cache if n.endswith("hook_pattern")]
    if not live_norm:
        hooks += [(n, freeze) for n in cache if n.endswith("hook_scale")]
    with torch.no_grad():
        logits = model.run_with_hooks(PROMPT, fwd_hooks=hooks)[0, -1]
    return float(logits[TARGET] - logits.mean())


def check_paths(model, graph, live_attention, live_norm, eps, rtol, atol):
    feats, acts = gm.feature_nodes(graph)
    pred = gm.total_effects(graph) * acts  # effect of a unit relative change, per feature
    deeper = [i for i, (l, _, _) in enumerate(feats.tolist()) if l < model.cfg.n_layers - 1]
    assert deeper, "the test needs features with downstream paths"
    for i in deeper:
        f = tuple(feats[i].tolist())
        up = twin_logit(model, f, eps, live_attention, live_norm)
        down = twin_logit(model, f, -eps, live_attention, live_norm)
        fd = (up - down) / (2 * eps)  # central difference
        assert fd == pytest.approx(float(pred[i]), rel=rtol, abs=atol), (i, feats[i].tolist())


def test_path_sums_equal_the_frozen_linear_twin(model):
    check_paths(model, graph_of(model), False, False, eps=1.0, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not HAS_FLAGS, reason="needs circuit-tracer with the PR #123 flags")
def test_path_sums_of_the_unfrozen_graph_are_first_order(model):
    g = graph_of(model, freeze_attention=False, freeze_layernorm=False)
    check_paths(model, g, True, True, eps=1e-2, rtol=2e-2, atol=2e-4)


def test_path_sums_include_indirect_paths(model):
    g = graph_of(model)
    assert not torch.allclose(
        gm.total_effects(g) * gm.feature_nodes(g)[1], gm.direct_contributions(g), atol=1e-4
    )


def test_total_effects_on_a_hand_made_chain():
    # features f0 -> f1 -> logit, and f0 -> logit directly; activations 2 and 4
    a = torch.zeros(4, 4)  # [f0, f1, err, logit]
    a[1, 0] = 2.0 * 3.0  # f0 moves f1's pre-activation by 3 per unit
    a[3, 1] = 4.0 * 5.0  # f1 moves the logit by 5 per unit
    a[3, 0] = 2.0 * 7.0  # f0 moves the logit by 7 per unit directly
    g = SimpleNamespace(
        adjacency_matrix=a,
        selected_features=torch.tensor([0, 1]),
        active_features=torch.tensor([[0, 1, 0], [1, 1, 0]]),
        activation_values=torch.tensor([2.0, 4.0]),
        logit_targets=[None],
        logit_probabilities=torch.tensor([1.0]),
    )
    assert gm.total_effects(g).tolist() == pytest.approx([7 + 3 * 5, 5])
    assert gm.direct_contributions(g).tolist() == pytest.approx([14, 20])
    g.activation_values = torch.tensor([2.0, 0.0])
    with pytest.raises(ValueError, match="zero activation"):
        gm.total_effects(g)


def test_faithfulness_curve_endpoints(model):
    g = graph_of(model)
    feats = [tuple(f) for f in gm.feature_nodes(g)[0].tolist()]
    order = list(np.argsort(-gm.node_influence(g).numpy()))
    with special_ids(model, [0]):
        c = iv.faithfulness_curve(model, PROMPT, feats, order, [0, len(feats)], TARGET, 3)
    keep, remove = iv.normalised(c)
    assert keep[0] == pytest.approx(0.0, abs=1e-5) and keep[-1] == pytest.approx(1.0, abs=1e-5)
    assert remove[0] == pytest.approx(0.0, abs=1e-5) and remove[-1] == pytest.approx(1.0, abs=1e-5)
    assert iv.log_area([1, 10, 100], np.array([0.0, 0.5, 1.0])) == pytest.approx(0.5)


def _sweep():
    import sys

    sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "experiments")
    )
    import circuits_sweep

    return circuits_sweep


def test_atp_is_the_real_models_first_order_effect(model):
    cs = _sweep()
    g = graph_of(model)
    feats = [tuple(f) for f in gm.feature_nodes(g)[0].tolist()]
    acts = gm.feature_nodes(g)[1].numpy()
    real = tiny_llama()
    real.reset_hooks(including_permanent=True)
    eps = 1e-2
    with special_ids(model, [0]):
        per_unit = cs.atp_per_unit(real, cs.atp_grads(real, PROMPT, TARGET), feats)
        for i in range(0, len(feats), max(1, len(feats) // 8)):
            f, a = feats[i], float(acts[i])
            up = iv.metrics(iv.last_logits(model, PROMPT, [(f, a * (1 + eps))]), TARGET)["m"]
            down = iv.metrics(iv.last_logits(model, PROMPT, [(f, a * (1 - eps))]), TARGET)["m"]
            fd = (up - down) / (2 * eps)
            assert fd == pytest.approx(a * per_unit[i], rel=3e-2, abs=1e-3), (i, f)


@pytest.mark.skipif(not HAS_FLAGS, reason="needs circuit-tracer with the PR #123 flags")
def test_sweep_one_prompt_end_to_end(model):
    cs = _sweep()
    real = tiny_llama()
    real.reset_hooks(including_permanent=True)
    p = {"pid": "t-00", "prompt": PROMPT, "answer_id": TARGET, "contrast_id": 3}
    args = SimpleNamespace(max_feature_nodes=None, batch_size=64, chunk=7)
    with special_ids(model, [0]):
        grads = cs.atp_grads(real, PROMPT, TARGET)
        res = cs.main_prompt(model, p, grads, args)
    c, d, g = res["curves"], res["dose"], res["graphs"]
    assert set(g.variant) == {"G1", "G2", "G3"}
    full = c[c.k == c.n_cand]
    assert len(full) and np.allclose(full.keep_m, full.full_m, atol=1e-4)
    assert np.allclose(full.remove_m, full.none_m, atol=1e-4)
    assert set(c.truth) == {"T1", "T2"} and c.ranking.str.startswith("prune_").any()
    assert set(d.truth) == {"T1", "T2"} and set(d.s) == set(cs.DOSE)
    ablated = d[(d.s == 0.0) & (d.truth == "T1")]
    assert np.isfinite(ablated[["pred_G1", "pred_G3", "pred_atp", "real_dm"]].to_numpy()).all()


def tiny_clt():
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

    base = tiny_llama()
    torch.manual_seed(1)
    clt = CrossLayerTranscoder(
        n_layers=base.cfg.n_layers,
        d_transcoder=32,
        d_model=base.cfg.d_model,
        lazy_decoder=False,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    for _, p in clt.named_parameters():
        nn.init.uniform_(p, a=-0.5, b=0.5)
    model = ReplacementModel.from_config(base.cfg, clt)
    model.load_state_dict(base.state_dict(), strict=False)
    return model


@pytest.mark.parametrize("kind", ["plt", "clt"])
@pytest.mark.parametrize("frozen", [False, True])
def test_batched_intervener_matches_feature_intervention(model, kind, frozen):
    from twinlens.ct.fast import Intervener

    m = model if kind == "plt" else tiny_clt()
    with special_ids(m, [0]):
        _, acts = m.get_activations(PROMPT)
        nz = acts.nonzero().tolist()
        feats = [tuple(x) for x in nz if x[1] > 0][:12]
        a0 = {f: float(acts[f]) for f in feats}
        fast = Intervener(m, PROMPT, feats, freeze_attention=frozen)
        rows = [
            [0.0] * len(feats),  # ablate everything
            [float("nan")] * len(feats),  # clean
            [a0[f] * 3 if i % 3 == 0 else float("nan") for i, f in enumerate(feats)],
            [0.0 if i < 5 else float("nan") for i in range(len(feats))],
        ]
        got = fast.run(torch.tensor(rows), chunk=3)
        for r, row in enumerate(rows):
            set_to = [(f, v) for f, v in zip(feats, row, strict=True) if v == v]
            want = iv.last_logits(m, PROMPT, set_to, freeze_attention=frozen)
            torch.testing.assert_close(got[r], want, rtol=1e-4, atol=1e-4)
