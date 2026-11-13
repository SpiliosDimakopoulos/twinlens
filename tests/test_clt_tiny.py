# ruff: noqa: F811  (the tiny-Gemma fixtures are imported, then requested by name)
"""Cross-layer transcoders on the tiny random Gemma-2 (no downloads).

The CLT is random and saved in circuit-tracer's on-disk format (``W_enc_i`` / ``W_dec_i``
safetensors, DECISIONS.md D5). Checks: every level reproduces the clean run, the linear levels
are affine, the Jacobian fixes are first-order expansions of the live levels, sources upstream
of the first edit are handled exactly, ``real`` is the decoder-direction edit at every output
layer, and a per-layer transcoder written as a CLT gives the old per-layer results.
"""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from test_gemma_tiny import (  # noqa: F401  (pytest fixtures)
    AFFINE_TOL,
    CLEAN_TOL,
    N_LAYERS,
    _strong,
    hf_and_path,
    model,
    tc_files,
    tokens,
    uncapped,
)

from twinlens.twin.replacement import LEVELS, REAL_MODES, FeatureEdit, Twin
from twinlens.twin.transcoders import CrossLayerTranscoderSet, GemmaScopeTranscoderSet

D_MODEL, D_SAE = 64, 128


def _save_clt(directory, enc, dec, dtype=torch.float32):
    """``enc[i]``: dict of W_enc_i [d_sae, d_model], b_enc_i, b_dec_i (+ threshold_i);
    ``dec[i]``: W_dec_i [d_sae, n_layers - i, d_model]."""
    for i in range(N_LAYERS):
        save_file(
            {k: v.to(dtype).contiguous() for k, v in enc[i].items()},
            directory / f"W_enc_{i}.safetensors",
        )
        save_file(
            {f"W_dec_{i}": dec[i].to(dtype).contiguous()}, directory / f"W_dec_{i}.safetensors"
        )
    return directory


@pytest.fixture(scope="module")
def clt_dir(tmp_path_factory):
    g = torch.Generator().manual_seed(5)
    enc, dec = {}, {}
    for i in range(N_LAYERS):
        enc[i] = {
            f"W_enc_{i}": torch.randn(D_SAE, D_MODEL, generator=g) * 0.02,
            f"b_enc_{i}": torch.randn(D_SAE, generator=g) * 0.3 - 0.2,
            f"b_dec_{i}": torch.randn(D_MODEL, generator=g) * 0.1,
        }
        dec[i] = torch.randn(D_SAE, N_LAYERS - i, D_MODEL, generator=g) * 0.1
    return _save_clt(tmp_path_factory.mktemp("clt"), enc, dec)


@pytest.fixture(scope="module")
def twin(model, clt_dir):
    return Twin(model, CrossLayerTranscoderSet(N_LAYERS, clt_dir))


def test_format_and_sources(twin, tokens):
    assert twin.cross and twin.tc_hook == "hook_resid_mid" and twin.skip_first
    assert twin.sources == {0: [0], 1: [0, 1], 2: [0, 1, 2]}
    trace = twin.trace(tokens)
    assert all(trace.dec[i].shape[1] == N_LAYERS - i for i in range(N_LAYERS))
    assert all((trace.acts[i][0] == 0).all() for i in range(N_LAYERS))  # BOS zeroed
    assert all(len(trace.active[i]) > 5 for i in range(N_LAYERS))


def test_dec_rows_keep_request_order(twin):
    src = twin.tcs[1]
    idx = torch.tensor([7, 2, 9, 2])
    rows = src.dec_rows(idx)
    assert rows.shape == (4, N_LAYERS - 1, D_MODEL)
    for r, f in zip(rows, idx.tolist(), strict=True):
        assert torch.equal(r, src.dec_rows([f])[0])


@pytest.mark.parametrize("mode", [*LEVELS, *REAL_MODES])
@pytest.mark.parametrize("layer", range(N_LAYERS))
def test_noop_edit_reproduces_clean_run(twin, tokens, mode, layer):
    """Hooks from ``layer`` on: sources below it use their clean contribution, the rest are
    recomputed; with a no-op edit every level must give the clean logits."""
    trace = twin.trace(tokens)
    pos, f, a = _strong(trace, layer)
    out = twin.run(trace, mode, [None, FeatureEdit(layer, pos, f, a)])
    torch.testing.assert_close(out[0], trace.logits[-1], **CLEAN_TOL)
    torch.testing.assert_close(out[1], trace.logits[-1], **CLEAN_TOL)


@pytest.mark.parametrize("mode", ["direct", "linear", "replacement", "linear_fulljac"])
def test_upstream_sources_need_no_hooks(twin, tokens, mode):
    """An edit at layer 1 run alone (sources at layer 0 taken from the clean trace) equals the
    same edit batched with a no-op at layer 0 (every source recomputed from layer 0)."""
    trace = twin.trace(tokens)
    pos, f, _ = _strong(trace, 1)
    p0, f0, a0 = _strong(trace, 0)
    edit = FeatureEdit(1, pos, f, 0.0)
    alone = twin.run(trace, mode, [edit])[0]
    batched = twin.run(trace, mode, [FeatureEdit(0, p0, f0, a0), edit])[1]
    torch.testing.assert_close(alone, batched, **CLEAN_TOL)


@pytest.mark.parametrize("mode", ["direct", "linear", "linear_lnjac", "linear_fulljac"])
def test_linear_levels_are_affine(twin, model, tokens, mode):
    with uncapped(model):
        trace = twin.trace(tokens)
        pos, f, a = _strong(trace, 0)
        clean = trace.logits[-1]
        full, half = twin.run(
            trace, mode, [FeatureEdit(0, pos, f, 0.0), FeatureEdit(0, pos, f, a / 2)]
        )
    torch.testing.assert_close(full - clean, 2 * (half - clean), **AFFINE_TOL)


def test_two_edits_add_up_in_linear_levels(twin, model, tokens):
    with uncapped(model):
        trace = twin.trace(tokens)
        clean = trace.logits[-1]
        # e1 sets (clamps) a layer-1 feature, which would cancel e0's effect on that feature;
        # placed one position earlier, e1's feature is out of e0's (causal) reach.
        e0 = FeatureEdit(0, *_strong(trace, 0)[:2], 0.0)
        p1 = trace.n_pos - 2
        e1 = FeatureEdit(1, p1, int(trace.acts[1][p1].argmax()), 0.0)
        for mode in ("linear", "linear_fulljac"):
            d0, d1 = (twin.run(trace, mode, [e])[0] - clean for e in (e0, e1))
            both = _run_both(twin, trace, mode, e0, e1) - clean
            torch.testing.assert_close(both, d0 + d1, **AFFINE_TOL)


def _run_both(twin, trace, mode, e0, e1):
    """Both edits on the same batch row (the public API puts one edit per row)."""
    from twinlens.twin.replacement import LEVELS as LV

    by_layer = {e0.layer: [(0, e0)], e1.layer: [(0, e1)]}
    hooks = twin._twin_hooks(trace, LV[mode], min(e0.layer, e1.layer), by_layer)
    with twin.model.hooks(hooks):
        return twin.model(trace.tokens, last_only=True)[0, -1]


def _rel_err(twin, model, tokens, jac, live):
    with uncapped(model):
        trace = twin.trace(tokens)
        pos, f, a = _strong(trace, 0)
        e = [FeatureEdit(0, pos, f, a * 0.99)]
        clean = trace.logits[-1]
        ref = twin.run(trace, live, e)[0] - clean
        approx = twin.run(trace, jac, e)[0] - clean
        base = twin.run(trace, "linear", e)[0] - clean
    return ((approx - ref).norm() / ref.norm()).item(), ((base - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize(
    ("jac", "live"), [("linear_lnjac", "linear_lnlive"), ("linear_fulljac", "linear_live")]
)
def test_jacobian_fixes_are_first_order(twin, model, tokens, jac, live):
    err, frozen = _rel_err(twin, model, tokens, jac, live)
    assert err < 0.05
    assert err < 0.5 * frozen


def test_real_writes_to_every_output_layer(twin, model, tokens):
    trace = twin.trace(tokens)
    pos, f, a = _strong(trace, 0)
    edit = FeatureEdit(0, pos, f, 0.0)
    rows = twin.tcs[0].dec_rows([f])[0]  # [n_layers, d_model]

    def add(l):
        def hook(out, _):
            out = out.clone()
            out[:, pos] += (0.0 - a) * rows[l]
            return out

        return hook

    with model.hooks([(f"blocks.{l}.hook_mlp_out", add(l)) for l in range(N_LAYERS)]):
        ref = model(trace.tokens, last_only=True)[0, -1]
    torch.testing.assert_close(twin.run(trace, "real", [edit])[0], ref, **CLEAN_TOL)
    # the cross-layer writes matter: dropping them changes the answer
    with model.hooks([("blocks.0.hook_mlp_out", add(0))]):
        own_only = model(trace.tokens, last_only=True)[0, -1]
    assert (own_only - ref).abs().max() > 1e-3


def test_real_frozen_attn_freezes_patterns(twin, model, tokens):
    trace = twin.trace(tokens)
    pos, f, _ = _strong(trace, 0)
    edit = [FeatureEdit(0, pos, f, 0.0)]
    real, frozen = twin.run(trace, "real", edit)[0], twin.run(trace, "real_frozen_attn", edit)[0]
    assert (real - frozen).abs().max() > 1e-4


def test_attribution_matches_direct_mode(twin, model, tokens):
    with uncapped(model):
        trace = twin.trace(tokens)
        token = int(trace.logits[-1].argmax())
        rows = twin.attribute_logit(trace, token, top=4)
        assert {r["layer"] for r in rows} - {N_LAYERS - 1}  # not only last-layer features
        for r in rows:
            edit = FeatureEdit(r["layer"], r["pos"], r["feature"], 0.0)
            delta = (twin.run(trace, "direct", [edit])[0, token] - trace.logits[-1, token]).item()
            assert delta == pytest.approx(-r["effect"], rel=2e-2, abs=2e-3)


def test_plt_written_as_clt_matches_per_layer_path(model, tc_files, tokens, tmp_path):
    """A Gemma Scope PLT stored as a CLT whose cross-layer decoders are zero gives the per-layer
    twin's results at every level (same input hook, threshold and BOS handling)."""
    enc, dec = {}, {}
    for i in range(N_LAYERS):
        z = np.load(tc_files[i])
        enc[i] = {
            f"W_enc_{i}": torch.from_numpy(z["W_enc"]).T,
            f"b_enc_{i}": torch.from_numpy(z["b_enc"]),
            f"b_dec_{i}": torch.from_numpy(z["b_dec"]),
            f"threshold_{i}": torch.from_numpy(z["threshold"]),
        }
        d = torch.zeros(z["W_dec"].shape[0], N_LAYERS - i, D_MODEL)
        d[:, 0] = torch.from_numpy(z["W_dec"])
        dec[i] = d
    clt = CrossLayerTranscoderSet(N_LAYERS, _save_clt(tmp_path, enc, dec))
    hook = "ln2.hook_normalized"
    as_clt = Twin(model, clt, tc_hook=hook, skip_first=False)
    plt = Twin(model, GemmaScopeTranscoderSet(range(N_LAYERS), hook, "hook_mlp_out", tc_files))
    tc_, tp = as_clt.trace(tokens), plt.trace(tokens)
    for i in range(N_LAYERS):
        assert torch.equal(tc_.active[i], tp.active[i])
    edits = [FeatureEdit(l, *_strong(tp, l)[:2], 0.0) for l in range(N_LAYERS)]
    for mode in [*LEVELS, *REAL_MODES]:
        torch.testing.assert_close(
            as_clt.run(tc_, mode, edits), plt.run(tp, mode, edits), atol=2e-4, rtol=2e-4
        )
    token = int(tp.logits[-1].argmax())
    ec, ep = as_clt.feature_effects(tc_, token), plt.feature_effects(tp, token)
    for i in range(N_LAYERS):
        torch.testing.assert_close(ec[i], ep[i], atol=1e-5, rtol=1e-4)


def test_bf16_files_load(model, clt_dir, tokens, tmp_path):
    """The released CLTs are BF16: rows are converted to F32 on read."""
    from twinlens.core.weights import MmapSafetensors

    for i in range(N_LAYERS):
        for kind in ("enc", "dec"):
            st = MmapSafetensors(clt_dir / f"W_{kind}_{i}.safetensors")
            save_file(
                {k: st.get(k).clone().to(torch.bfloat16) for k in st.keys()},  # noqa: SIM118 (not a dict)
                tmp_path / f"W_{kind}_{i}.safetensors",
            )
    twin = Twin(model, CrossLayerTranscoderSet(N_LAYERS, tmp_path))
    trace = twin.trace(tokens)
    assert trace.dec[0].dtype == torch.float32
    out = twin.run(trace, "linear_fulljac", [FeatureEdit(0, *_strong(trace, 0)[:2], 0.0)])
    assert torch.isfinite(out).all()
