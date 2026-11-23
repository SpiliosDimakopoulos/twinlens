"""Attribution patching must be the real model's full first-order effect.

The first version of ``experiments/atp_baseline.py`` detached every MLP output, which cut all
paths through later MLPs (found 2026-11-24). This pins the fix: the gradient at layer 0 must
predict a small edit of layer 0's MLP output in the real model.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from atp_baseline import mlp_out_grads  # noqa: E402

TEXT = "When Mary and John went to the store, John gave a drink to"


def _logp(model, toks, target, layer, pos, delta) -> float:
    def edit(x, _):
        x = x.clone()
        x[0, pos] += delta
        return x

    with torch.no_grad(), model.hooks([(f"blocks.{layer}.hook_mlp_out", edit)]):
        return float(model(toks)[0, -1].log_softmax(-1)[target])


def test_atp_gradient_matches_a_finite_difference(model):
    """Move each MLP output along its own gradient: the change in log p must equal |grad| · eps.
    The old, truncated gradient points elsewhere at early layers and fails this by far."""
    target, grads = mlp_out_grads(model, TEXT)
    toks = model.to_tokens(TEXT)
    eps = 0.1
    for layer, pos in ((0, 4), (2, -1), (5, 7), (10, -1)):
        g = grads[layer][pos]
        v = eps * g / g.norm()
        fd = _logp(model, toks, target, layer, pos, v) - _logp(model, toks, target, layer, pos, -v)
        fd /= 2 * eps
        assert abs(fd - float(g.norm())) <= 0.05 * float(g.norm()), (layer, pos, fd, g.norm())
