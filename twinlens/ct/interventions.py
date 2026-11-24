"""Real interventions through circuit-tracer's ``feature_intervention`` (DECISIONS.md D10).

With no ``constrained_layers``, ``feature_intervention`` adds ``Δa · W_dec`` to the real MLP
output and recomputes everything downstream. With ``freeze_attention=False`` (T1) attention and
LayerNorm are live; with the default ``freeze_attention=True`` (T2) attention patterns stay at
their clean values and LayerNorm is still live.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

Feature = tuple[int, int, int]  # (layer, pos, feature_idx)


def last_logits(
    model,
    tokens: torch.Tensor,
    set_to: Sequence[tuple[Feature, float]] = (),
    freeze_attention: bool = False,
) -> torch.Tensor:
    """Logits at the last position after setting each feature to its value (none = clean run).

    Always goes through ``feature_intervention`` so that both circuit-tracer backends behave
    the same; with no interventions it is the clean forward pass."""
    iv = [(int(l), int(p), int(f), float(v)) for (l, p, f), v in set_to]
    with torch.no_grad():
        logits, _ = model.feature_intervention(
            tokens, iv, freeze_attention=freeze_attention, return_activations=False
        )
    return logits.reshape(-1, logits.shape[-1])[-1].float()


def metrics(logits: torch.Tensor, answer: int, contrast: int | None = None) -> dict[str, float]:
    """``m``: demeaned answer logit (circuit-tracer's target). ``dld``: answer − contrast."""
    out = {"m": float(logits[answer] - logits.mean())}
    if contrast is not None:
        out["dld"] = float(logits[answer] - logits[contrast])
    return out


def ablate(model, tokens, feats: Sequence[Feature], **kw) -> torch.Tensor:
    return last_logits(model, tokens, [(f, 0.0) for f in feats], **kw)


def faithfulness_curve(
    model,
    tokens: torch.Tensor,
    candidates: Sequence[Feature],
    order: Sequence[int],
    ks: Sequence[int],
    answer: int,
    contrast: int | None = None,
    freeze_attention: bool = False,
    reference: tuple[dict, dict] | None = None,
) -> dict:
    """keep(K) / remove(K) curves of D10-A for one ranking.

    ``order`` lists indices into ``candidates``, most important first. keep(K) zeroes every
    candidate outside the top K; remove(K) zeroes the top K. ``reference`` = (metrics of the
    clean run, metrics with every candidate zeroed), computed once per prompt and truth.
    """
    if reference is None:
        reference = (
            metrics(last_logits(model, tokens), answer, contrast),
            metrics(
                ablate(model, tokens, candidates, freeze_attention=freeze_attention),
                answer,
                contrast,
            ),
        )
    full, none = reference
    out: dict = {"k": [], "keep": [], "remove": []}
    for k in ks:
        top = [candidates[i] for i in order[:k]]
        rest = [candidates[i] for i in order[k:]]
        out["k"].append(int(k))
        out["keep"].append(
            metrics(
                ablate(model, tokens, rest, freeze_attention=freeze_attention), answer, contrast
            )
        )
        out["remove"].append(
            metrics(ablate(model, tokens, top, freeze_attention=freeze_attention), answer, contrast)
        )
    out["full"], out["none"] = full, none
    return out


def normalised(curve: dict, key: str = "m") -> tuple[np.ndarray, np.ndarray]:
    """``(F_keep, F_remove)`` of a curve: 0 = all candidates zeroed, 1 = clean."""
    full, none = curve["full"][key], curve["none"][key]
    span = full - none
    keep = np.array([(x[key] - none) / span for x in curve["keep"]])
    remove = np.array([(full - x[key]) / span for x in curve["remove"]])
    return keep, remove


def log_area(ks: Sequence[int], f: np.ndarray) -> float:
    """Area under ``f`` over log10 K (trapezoid), normalised to the width of the grid."""
    x = np.log10(np.asarray(ks, dtype=float))
    if len(x) < 2:
        return float(f[0])
    return float(np.trapezoid(f, x) / (x[-1] - x[0]))
