"""Measurements behind the reproduction suite (published GPT-2 small circuits)."""

from __future__ import annotations

import torch

from twinlens.causal.attribution import head_dla_grid
from twinlens.causal.interventions import HeadAblation, build_hooks, mean_cache_for
from twinlens.circuits.datasets import IOIBatch, repeated_random_tokens
from twinlens.core.gpt2 import GPT2

# Heads reported in the literature for GPT-2 small (layer, head).
INDUCTION_HEADS = {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}  # Olsson et al. 2022; Wang et al. 2022
NAME_MOVER_HEADS = {(9, 9), (9, 6), (10, 0)}  # Wang et al. 2022
NEGATIVE_NAME_MOVERS = {(10, 7), (11, 10)}  # Wang et al. 2022


def induction_scores(model: GPT2, seq_len: int = 50, n_batch: int = 8) -> torch.Tensor:
    """[n_layers, n_heads] mean attention from token i in the 2nd repeat to token i-seq_len+1."""
    tokens = repeated_random_tokens(model, n_batch, seq_len)
    names = [f"blocks.{l}.attn.hook_pattern" for l in range(model.cfg.n_layers)]
    _, cache = model.run_with_cache(tokens, names, last_only=True)
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer, name in enumerate(names):
        pat = cache[name]  # [b, h, q, k]
        stripe = pat.diagonal(offset=-(seq_len - 1), dim1=-2, dim2=-1)  # [b, h, seq_len+1]
        scores[layer] = stripe[..., 1:].mean((0, -1))
    return scores


def ioi_logit_diff(model: GPT2, batch: IOIBatch, hooks: list | None = None) -> float:
    with model.hooks(hooks or []):
        last = model(batch.tokens)[:, -1]
    idx = torch.arange(len(batch.io))
    return (last[idx, batch.io] - last[idx, batch.s]).mean().item()


def ioi_head_dla(model: GPT2, batch: IOIBatch) -> torch.Tensor:
    direction = model.W_E[batch.io] - model.W_E[batch.s]  # tied unembedding, [batch, d_model]
    return head_dla_grid(model, batch.tokens, direction)


def ioi_ablate_heads(model: GPT2, batch: IOIBatch, heads: set[tuple[int, int]]) -> float:
    """Logit diff after mean-ablating ``heads`` (means taken over the ABC distribution)."""
    names = sorted({f"blocks.{l}.attn.hook_z" for l, _ in heads})
    means = mean_cache_for(model, batch.corrupt_tokens, names)
    specs = [HeadAblation(layer=l, head=h, mode="mean") for l, h in heads]
    return ioi_logit_diff(model, batch, build_hooks(model, specs, means))


def top_k(grid: torch.Tensor, k: int, largest: bool = True) -> list[tuple[int, int]]:
    flat = grid.flatten().topk(k, largest=largest).indices
    return [(int(i) // grid.shape[1], int(i) % grid.shape[1]) for i in flat]
