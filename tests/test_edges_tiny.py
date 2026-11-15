# ruff: noqa: F811  (the tiny-Gemma fixtures are imported, then requested by name)
"""Feature-to-feature edges (``fidelity/edges.py``, DECISIONS.md D8) on the tiny random
Gemma-2, with a per-layer and a cross-layer transcoder (no downloads)."""

import pytest
import torch
from test_clt_tiny import clt_dir  # noqa: F401  (pytest fixture)
from test_gemma_tiny import (  # noqa: F401  (pytest fixtures)
    N_LAYERS,
    hf_and_path,
    model,
    tc_files,
    tokens,
    uncapped,
)

from twinlens.fidelity.edges import LEVELS, TRUTHS, score_edits
from twinlens.fidelity.engine import EditSpec
from twinlens.twin.replacement import FeatureEdit, Twin
from twinlens.twin.transcoders import CrossLayerTranscoderSet, GemmaScopeTranscoderSet

COLS = [f"{m}_dact" for m in (*LEVELS, *TRUTHS)] + [f"{m}_dpre" for m in TRUTHS]


@pytest.fixture(scope="module", params=["plt", "clt"])
def twin(request, model, tc_files, clt_dir):
    if request.param == "clt":
        return Twin(model, CrossLayerTranscoderSet(N_LAYERS, clt_dir))
    tcs = GemmaScopeTranscoderSet(range(N_LAYERS), "ln2.hook_normalized", "hook_mlp_out", tc_files)
    return Twin(model, tcs)


def _edit(trace, scale):
    p = trace.n_pos - 4
    f = int(trace.acts[0][p].argmax())
    a = trace.acts[0][p, f].item()
    return [EditSpec(FeatureEdit(0, p, f, scale * a), "ablate", "top", 0.0)]


def test_noop_edit_changes_nothing(twin, tokens):
    trace = twin.trace(tokens)
    rows = score_edits(twin, trace, _edit(trace, 1.0), k_top=5, k_random=5)
    assert rows
    for r in rows:
        for c in COLS:
            assert abs(r[c]) < 1e-4, (c, r[c])


def test_targets_are_downstream_active_and_ranked(twin, tokens):
    trace = twin.trace(tokens)
    rows = score_edits(twin, trace, _edit(trace, 0.0), k_top=5, k_random=5)
    for r in rows:
        assert r["target_layer"] > r["layer"]
        assert r["target_pos"] >= r["pos"] and r["target_pos"] > 0
        assert r["act0"] > 0
        act = trace.acts[r["target_layer"]][r["target_pos"], r["target_feature"]].item()
        assert r["act0"] == pytest.approx(act, rel=1e-5)
    top = [abs(r["linear_dact"]) for r in rows if r["target_selection"] == "top"]
    rand = [abs(r["linear_dact"]) for r in rows if r["target_selection"] == "random"]
    assert len(top) == 5 and 0 < len(rand) <= 5
    assert min(top) >= max(rand)


def test_linear_levels_are_affine_in_the_edit(twin, model, tokens):
    with uncapped(model):
        trace = twin.trace(tokens)
        full = score_edits(twin, trace, _edit(trace, 0.0), k_top=5, k_random=5)
        half = score_edits(twin, trace, _edit(trace, 0.5), k_top=5, k_random=5)
    key = ("target_layer", "target_pos", "target_feature")
    assert [tuple(r[k] for k in key) for r in full] == [tuple(r[k] for k in key) for r in half]
    for a, b in zip(full, half, strict=True):
        for m in LEVELS:
            assert a[f"{m}_dact"] == pytest.approx(2 * b[f"{m}_dact"], rel=1e-3, abs=1e-5)


def test_real_truth_is_the_encoder_on_the_real_input(twin, tokens):
    """``real_dpre`` equals the target's encoder applied to the real model's transcoder input
    with the edit, read from an independent capture."""
    trace = twin.trace(tokens)
    specs = _edit(trace, 0.0)
    rows = score_edits(twin, trace, specs, k_top=3, k_random=3)
    cap: dict = {}
    twin.run(trace, "real", [specs[0].edit], capture=cap)
    for r in rows:
        l, p, f = r["target_layer"], r["target_pos"], r["target_feature"]
        W, b = twin.tcs[l].enc_rows(torch.tensor([f]))
        pre = float(cap[l][0, p] @ W[0] + b[0])
        assert r["real_dpre"] == pytest.approx(pre - r["pre0"], abs=1e-5)
        # the nonlinearity: either the target stays on (dact = dpre) or it turns off
        assert r["real_dact"] == pytest.approx(r["real_dpre"], abs=1e-5) or r[
            "real_dact"
        ] == pytest.approx(-r["act0"], abs=1e-5)
