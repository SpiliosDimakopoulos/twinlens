"""Direct logit attribution (DLA) and logit lens on the real model.

DLA decomposes the final residual stream at one position into the contributions of every
component (embeddings, each attention head, each MLP). Each contribution is projected onto an
unembedding direction through the *frozen* final LayerNorm scale. Because the projection is
linear, the contributions sum exactly to the logit (up to the LN bias term, which is returned
separately).
"""

from __future__ import annotations

import torch

from twinlens.core.gpt2 import GPT2


def component_outputs(
    model: GPT2, tokens: torch.Tensor, pos: int = -1
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Return (labels, outputs [n_comp, batch, d_model], ln_final scale [batch, 1]) at ``pos``."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    names = ["hook_embed", "hook_pos_embed", "ln_final.hook_scale"]
    for layer in range(L):
        names += [f"blocks.{layer}.attn.hook_result", f"blocks.{layer}.hook_mlp_out"]
    _, cache = model.run_with_cache(tokens, names, last_only=True)
    labels, outs = (
        ["embed", "pos_embed"],
        [cache["hook_embed"][:, pos], cache["hook_pos_embed"][:, pos]],
    )
    for layer in range(L):
        res = cache[f"blocks.{layer}.attn.hook_result"][:, pos]  # [batch, head, d_model]
        for h in range(H):
            labels.append(f"L{layer}H{h}")
            outs.append(res[:, h])
        labels.append(f"L{layer}.attn_bias")
        outs.append(model.blocks[layer]["b_O"].expand_as(res[:, 0]))
        labels.append(f"L{layer}.mlp")
        outs.append(cache[f"blocks.{layer}.hook_mlp_out"][:, pos])
    return labels, torch.stack(outs), cache["ln_final.hook_scale"][:, pos]


def direct_logit_attribution(
    model: GPT2, tokens: torch.Tensor, direction: torch.Tensor, pos: int = -1
) -> tuple[list[str], torch.Tensor]:
    """Contribution of each component to ``direction · ln_final(resid)``, averaged over the batch.

    ``direction`` is [d_model] or [batch, d_model] (e.g. W_U[:, io] - W_U[:, s]).
    """
    labels, outs, scale = component_outputs(model, tokens, pos)
    centered = outs - outs.mean(-1, keepdim=True)  # LN centring is linear
    gamma_dir = direction * model.ln_f[0]  # fold the LN weight into the direction
    if gamma_dir.dim() == 1:
        gamma_dir = gamma_dir.expand(outs.shape[1], -1)
    contrib = torch.einsum("cbd,bd->cb", centered, gamma_dir) / scale[None, :, 0]
    return labels, contrib.mean(1)


def head_dla_grid(
    model: GPT2, tokens: torch.Tensor, direction: torch.Tensor, pos: int = -1
) -> torch.Tensor:
    """[n_layers, n_heads] DLA per head."""
    labels, contrib = direct_logit_attribution(model, tokens, direction, pos)
    grid = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for lab, c in zip(labels, contrib, strict=False):
        if lab.startswith("L") and "H" in lab:
            layer, head = lab[1:].split("H")
            grid[int(layer), int(head)] = c
    return grid


def logit_lens(model: GPT2, tokens: torch.Tensor, k: int = 5) -> dict[str, torch.Tensor]:
    """Top-k next-token predictions decoded from the residual stream after every layer.

    Returns ``top_ids``/``top_probs`` of shape [n_layers + 1, pos, k] (row 0 = embeddings) and
    ``final_rank`` [n_layers + 1, pos]: the rank of the model's final prediction at each layer.
    """
    names = ["blocks.0.hook_resid_pre"] + [
        f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)
    ]
    logits, cache = model.run_with_cache(tokens[:1], names)
    final = logits[0].argmax(-1)  # [pos]
    top_ids, top_probs, ranks, fprobs = [], [], [], []
    for n in names:  # one layer at a time: [pos, vocab] instead of [layers, pos, vocab]
        lens_logits = model.unembed(model.ln_final(cache[n][0]))
        probs = lens_logits.softmax(-1)
        top = probs.topk(k, dim=-1)
        top_ids.append(top.indices)
        top_probs.append(top.values)
        final_logit = lens_logits.gather(-1, final[:, None])
        ranks.append((lens_logits > final_logit).sum(-1))
        fprobs.append(probs.gather(-1, final[:, None])[:, 0])
    return {
        "top_ids": torch.stack(top_ids),
        "top_probs": torch.stack(top_probs),
        "final_rank": torch.stack(ranks),
        "final_prob": torch.stack(fprobs),
    }
