"""Reliability signals: cheap features of an edit, computed from the clean run only.

They are what the twin can know *before* the real model is consulted. The fidelity study asks
whether they predict when the twin's prediction will be wrong (see ``trust.py``).
"""

from __future__ import annotations

from twinlens.twin.replacement import CleanTrace, FeatureEdit, Twin

SIGNALS = (
    "layers_after",
    "pos_from_end",
    "activation",
    "rel_edit_size",
    "abs_direct_effect",
    "downstream_err_frac",
    "attn_to_pos",
    "n_active_downstream",
    "clean_top1_prob",
)


def reliability_signals(
    twin: Twin, trace: CleanTrace, edit: FeatureEdit, direct_effect: float
) -> dict[str, float]:
    L = twin.model.cfg.n_layers
    last = trace.n_pos - 1
    a = trace.acts[edit.layer][edit.pos, edit.feature].item()
    dec = twin.dec_row(trace, edit.layer, edit.feature)
    resid = trace.resid_mid[edit.layer][0, edit.pos]
    downstream = range(edit.layer + 1, L)
    err_frac = [
        (trace.error[l][0, last].norm() / (trace.mlp_out[l][0, last].norm() + 1e-6)).item()
        for l in downstream
    ]
    # The twin freezes attention patterns, so heavy reliance on attention from the last
    # position to the edited one (after the edit) is a candidate risk factor.
    attn = [trace.patterns[l][0, :, last, edit.pos].mean().item() for l in downstream]
    return {
        "layer": float(edit.layer),
        "layers_after": float(L - 1 - edit.layer),
        "pos_from_end": float(last - edit.pos),
        "activation": a,
        "delta_act": edit.value - a,
        "rel_edit_size": (abs(edit.value - a) * dec.norm() / resid.norm()).item(),
        "direct_effect": direct_effect,
        "abs_direct_effect": abs(direct_effect),
        "downstream_err_frac": sum(err_frac) / len(err_frac) if err_frac else 0.0,
        "attn_to_pos": sum(attn) / len(attn) if attn else 0.0,
        "n_active_downstream": float(sum(int((trace.acts[l][last] > 0).sum()) for l in downstream)),
        "clean_top1_prob": trace.logits[-1].softmax(-1).max().item(),
    }
