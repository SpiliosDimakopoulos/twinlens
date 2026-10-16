"""Activation patching and causal tracing.

* ``patch_resid``: patch clean activations into a corrupted run, one (layer, position) at a time
  (Meng et al., 2022; Wang et al., 2022). Returns the normalised metric recovery.
* ``patch_heads``: the same, one attention head at a time (all positions).
* ``causal_trace``: ROME-style tracing. The subject embeddings are corrupted with Gaussian noise,
  then a single hidden state (residual, MLP or attention output) is restored at each
  (layer, position).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from twinlens.core.gpt2 import GPT2

Metric = Callable[[torch.Tensor], torch.Tensor]  # logits -> scalar


def logit_diff_metric(correct: torch.Tensor, wrong: torch.Tensor, pos: int = -1) -> Metric:
    """Mean over the batch of logit[correct] - logit[wrong] at ``pos``."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, pos]
        idx = torch.arange(last.shape[0])
        return (last[idx, correct] - last[idx, wrong]).mean()

    return metric


def _normalise(values: torch.Tensor, clean: float, corrupt: float) -> torch.Tensor:
    return (values - corrupt) / (clean - corrupt + 1e-9)


def patch_resid(
    model: GPT2,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    metric: Metric,
    hook: str = "hook_resid_pre",
) -> torch.Tensor:
    """[n_layers, n_pos] fraction of the clean-vs-corrupt metric gap recovered by patching.

    All positions of a layer are patched in one batched forward pass.
    """
    b, n_pos = clean.shape
    names = [f"blocks.{layer}.{hook}" for layer in range(model.cfg.n_layers)]
    clean_logits, clean_cache = model.run_with_cache(clean, names)
    clean_m = metric(clean_logits).item()
    corrupt_m = metric(model(corrupt, last_only=True)).item()
    rows = corrupt.repeat(n_pos, 1)
    out = torch.zeros(model.cfg.n_layers, n_pos)
    for layer, name in enumerate(names):

        def fn(x: torch.Tensor, _: str, name: str = name) -> torch.Tensor:
            x = x.clone()
            for pos in range(n_pos):
                x[pos * b : (pos + 1) * b, pos] = clean_cache[name][:, pos]
            return x

        with model.hooks([(name, fn)]):
            logits = model(rows, last_only=True)
        for pos in range(n_pos):
            out[layer, pos] = metric(logits[pos * b : (pos + 1) * b])
    return _normalise(out, clean_m, corrupt_m)


def patch_heads(
    model: GPT2, clean: torch.Tensor, corrupt: torch.Tensor, metric: Metric
) -> torch.Tensor:
    """[n_layers, n_heads] metric recovery from patching each head's output (hook_z).

    All heads of a layer are patched in a single batched forward pass.
    """
    H, b = model.cfg.n_heads, clean.shape[0]
    names = [f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)]
    clean_logits, clean_cache = model.run_with_cache(clean, names)
    clean_m = metric(clean_logits).item()
    corrupt_m = metric(model(corrupt, last_only=True)).item()
    rows = corrupt.repeat(H, 1)  # row block h patches head h
    out = torch.zeros(model.cfg.n_layers, H)
    for layer, name in enumerate(names):

        def fn(z: torch.Tensor, _: str, name: str = name) -> torch.Tensor:
            z = z.clone()
            for h in range(H):
                z[h * b : (h + 1) * b, :, h] = clean_cache[name][:, :, h]
            return z

        with model.hooks([(name, fn)]):
            logits = model(rows, last_only=True)
        for h in range(H):
            out[layer, h] = metric(logits[h * b : (h + 1) * b])
    return _normalise(out, clean_m, corrupt_m)


def causal_trace(
    model: GPT2,
    tokens: torch.Tensor,
    subject_positions: list[int],
    target: int,
    kind: str = "resid",
    noise_scale: float = 3.0,
    n_samples: int = 5,
    window: int = 1,
    seed: int = 0,
) -> dict[str, torch.Tensor | float]:
    """ROME causal tracing for one prompt (``tokens`` of shape [1, pos]).

    Returns ``effect`` [n_layers, n_pos]: the probability of ``target`` when one state is restored
    in the corrupted run, together with the clean and corrupted probabilities. ``kind`` is
    "resid" | "mlp" | "attn". For mlp/attn, a window of ``window`` layers is restored
    (Meng et al. use 10 for GPT-2 XL).
    """
    hook = {"resid": "hook_resid_post", "mlp": "hook_mlp_out", "attn": "hook_attn_out"}[kind]
    n_layers, n_pos = model.cfg.n_layers, tokens.shape[1]
    names = [f"blocks.{layer}.{hook}" for layer in range(n_layers)]
    clean_logits, clean_cache = model.run_with_cache(tokens, names)
    p_clean = clean_logits[0, -1].softmax(-1)[target].item()

    gen = torch.Generator().manual_seed(seed)
    emb_std = model.W_E.std().item()
    noise = torch.randn(n_samples, len(subject_positions), model.cfg.d_model, generator=gen)
    noise = noise * noise_scale * emb_std
    batch = tokens.expand(n_samples, -1)

    def corrupt(x: torch.Tensor, _: str) -> torch.Tensor:
        x = x.clone()
        x[:, subject_positions] += noise
        return x

    with model.hooks([("hook_embed", corrupt)]):
        p_corrupt = model(batch, last_only=True)[:, -1].softmax(-1)[:, target].mean().item()

    # Batch layout: row block ``pos`` restores position ``pos`` (n_samples rows each).
    rows = batch.repeat(n_pos, 1)
    big_noise = noise.repeat(n_pos, 1, 1)

    def corrupt_rows(x: torch.Tensor, _: str) -> torch.Tensor:
        x = x.clone()
        x[:, subject_positions] += big_noise
        return x

    effect = torch.zeros(n_layers, n_pos)
    for layer in range(n_layers):
        if kind == "resid":
            layers = [layer]
        else:
            lo = max(0, layer - window // 2)
            layers = list(range(lo, min(n_layers, lo + window)))
        restore = []
        for lay in layers:
            name = names[lay]

            def fn(x: torch.Tensor, _: str, name: str = name) -> torch.Tensor:
                x = x.clone()
                for pos in range(n_pos):
                    x[pos * n_samples : (pos + 1) * n_samples, pos] = clean_cache[name][0, pos]
                return x

            restore.append((name, fn))
        with model.hooks([("hook_embed", corrupt_rows), *restore]):
            probs = model(rows, last_only=True)[:, -1].softmax(-1)[:, target]
        effect[layer] = probs.view(n_pos, n_samples).mean(1)
    return {"effect": effect, "p_clean": p_clean, "p_corrupt": p_corrupt}
