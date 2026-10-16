"""Per-layer MLP transcoders for GPT-2 small (Dunefsky, Chlenski & Nanda, 2024).

Convention (verified in DECISIONS.md, D2): each transcoder reads ``LayerNormPre(resid_mid)``,
i.e. the residual stream centred and divided by its LN scale without the affine weights, and
predicts ``hook_mlp_out``:

    acts = relu(x @ W_enc + b_enc)          x = (resid_mid - mean) / scale
    mlp_out ≈ acts @ W_dec + b_dec
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from twinlens.core.weights import MmapSafetensors, hf_file

REPO = "jacobdunefsky/gpt2small-transcoders"


@dataclass
class Transcoder:
    layer: int
    W_enc: torch.Tensor  # [d_model, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    W_dec: torch.Tensor  # [d_sae, d_model]
    b_dec: torch.Tensor  # [d_model]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    def pre(self, x_normed: torch.Tensor) -> torch.Tensor:
        return x_normed @ self.W_enc + self.b_enc

    def encode(self, x_normed: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.pre(x_normed))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec


class TranscoderSet:
    """Lazily memory-maps one transcoder per layer."""

    def __init__(self, n_layers: int = 12):
        self.n_layers = n_layers
        self._cache: dict[int, Transcoder] = {}

    def __getitem__(self, layer: int) -> Transcoder:
        if layer not in self._cache:
            st = MmapSafetensors(hf_file(REPO, f"gpt2-small-dun-chl-mlp-tc{layer}/sae.safetensors"))
            self._cache[layer] = Transcoder(
                layer, st.get("W_enc"), st.get("b_enc"), st.get("W_dec"), st.get("b_dec")
            )
        return self._cache[layer]

    def __iter__(self):
        return (self[layer] for layer in range(self.n_layers))


def layernorm_pre(resid: torch.Tensor, scale: torch.Tensor | None = None, eps: float = 1e-5):
    """Centre and scale like LayerNorm without the affine. ``scale`` can be frozen."""
    x = resid - resid.mean(-1, keepdim=True)
    if scale is None:
        scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return x / scale
