"""A hookable Qwen3 (RMSNorm, GQA, RoPE, QK-norm, SwiGLU) in plain PyTorch.

Same hook names and interface as ``GPT2`` so the twin, the fidelity engine and the API work
unchanged. Differences from GPT-2 that the hooks expose:

    every RMSNorm has ``.hook_scale`` and ``.hook_normalized`` (x / rms, before the weight):
    blocks.{l}.ln1 / ln2, blocks.{l}.attn.q_norm / k_norm (per head), ln_final
    attn.hook_q / hook_k are taken after QK-norm and RoPE; hook_k / hook_v are expanded from the
    8 KV heads to the 16 query heads (GQA), so all attention hooks are [batch, pos, 16, 128].

Weights are a memory-mapped F32 copy of the BF16 checkpoint (see ``weights.convert_to_f32``).
No BOS is prepended (Qwen convention).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from twinlens.core.gpt2 import GPT2, Tokenizer
from twinlens.core.weights import CACHE_DIR, MmapSafetensors, convert_to_f32, hf_file

REPO = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class QwenConfig:
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    d_model: int = 1024
    d_head: int = 128
    d_mlp: int = 3072
    d_vocab: int = 151936
    n_ctx: int = 4096
    ln_eps: float = 1e-6
    rope_theta: float = 1_000_000.0


class Qwen3(GPT2):
    """Hookable Qwen3-0.6B forward pass (inference only)."""

    def __init__(self, weights: MmapSafetensors, cfg: QwenConfig, tokenizer: Tokenizer):
        torch.set_grad_enabled(False)
        self.cfg = cfg  # type: ignore[assignment]
        self.tokenizer = tokenizer
        w = weights.get
        self.W_E = w("model.embed_tokens.weight")  # [vocab, d]; tied unembedding
        self.ln_f = (w("model.norm.weight"), torch.zeros(cfg.d_model))
        self.blocks = []
        for layer in range(cfg.n_layers):
            p = f"model.layers.{layer}."
            self.blocks.append(
                {
                    "ln1": w(p + "input_layernorm.weight"),
                    "ln2": w(p + "post_attention_layernorm.weight"),
                    "W_Q": w(p + "self_attn.q_proj.weight"),  # [H*dh, d]
                    "W_K": w(p + "self_attn.k_proj.weight"),  # [Hkv*dh, d]
                    "W_V": w(p + "self_attn.v_proj.weight"),
                    "W_O": w(p + "self_attn.o_proj.weight"),  # [d, H*dh]
                    "q_norm": w(p + "self_attn.q_norm.weight"),
                    "k_norm": w(p + "self_attn.k_norm.weight"),
                    "W_gate": w(p + "mlp.gate_proj.weight"),
                    "W_up": w(p + "mlp.up_proj.weight"),
                    "W_down": w(p + "mlp.down_proj.weight"),
                }
            )
        inv = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, dtype=torch.float64) / cfg.d_head)
        )
        self._inv_freq = inv.float()
        self._hooks = defaultdict(list)

    @classmethod
    def load(cls, name: str = "qwen3-0.6b") -> Qwen3:  # type: ignore[override]
        src = hf_file(REPO, "model.safetensors")
        f32 = convert_to_f32(src, CACHE_DIR / "twinlens" / "qwen3-0.6b.f32.safetensors")
        return cls(MmapSafetensors(f32), QwenConfig(), Tokenizer(hf_file(REPO, "tokenizer.json")))

    def hook_names(self) -> list[str]:
        names = ["hook_embed", "ln_final.hook_scale", "ln_final.hook_normalized"]
        for layer in range(self.cfg.n_layers):
            b = f"blocks.{layer}."
            names += [
                b + s
                for s in (
                    "hook_resid_pre",
                    "ln1.hook_scale",
                    "ln1.hook_normalized",
                    "attn.q_norm.hook_scale",
                    "attn.q_norm.hook_normalized",
                    "attn.k_norm.hook_scale",
                    "attn.k_norm.hook_normalized",
                    "attn.hook_q",
                    "attn.hook_k",
                    "attn.hook_v",
                    "attn.hook_pattern",
                    "attn.hook_z",
                    "attn.hook_result",
                    "hook_attn_out",
                    "hook_resid_mid",
                    "ln2.hook_scale",
                    "ln2.hook_normalized",
                    "hook_mlp_in",
                    "mlp.hook_post",
                    "hook_mlp_out",
                    "hook_resid_post",
                )
            ]
        return names

    def _rms(self, x: torch.Tensor, weight: torch.Tensor, name: str) -> torch.Tensor:
        scale = self._h(name, (x.pow(2).mean(-1, keepdim=True) + self.cfg.ln_eps).sqrt())
        normed = self._h(name.replace("hook_scale", "hook_normalized"), x / scale)
        return normed * weight

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], dtype=torch.float32)
        ang = pos[:, None] * self._inv_freq[None]  # [pos, dh/2]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None]
        half = x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * cos + rot * sin

    def forward(
        self, tokens: torch.Tensor, stop_at_layer: int | None = None, last_only: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens[None]
        cfg = self.cfg
        B, P = tokens.shape
        H, Hkv, dh = cfg.n_heads, cfg.n_kv_heads, cfg.d_head
        resid = self._h("hook_embed", self.W_E[tokens])
        mask = torch.ones(P, P, dtype=torch.bool).tril()
        for layer, blk in enumerate(self.blocks[:stop_at_layer]):
            b = f"blocks.{layer}."
            resid = self._h(b + "hook_resid_pre", resid)
            x = self._rms(resid, blk["ln1"], b + "ln1.hook_scale")
            q = (x @ blk["W_Q"].T).view(B, P, H, dh)
            k = (x @ blk["W_K"].T).view(B, P, Hkv, dh)
            v = (x @ blk["W_V"].T).view(B, P, Hkv, dh)
            q = self._rms(q, blk["q_norm"], b + "attn.q_norm.hook_scale")
            k = self._rms(k, blk["k_norm"], b + "attn.k_norm.hook_scale")
            rep = H // Hkv
            q = self._h(b + "attn.hook_q", self._rope(q))
            k = self._h(b + "attn.hook_k", self._rope(k).repeat_interleave(rep, dim=2))
            v = self._h(b + "attn.hook_v", v.repeat_interleave(rep, dim=2))
            scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(dh)
            scores = scores.masked_fill(~mask, float("-inf"))
            pattern = self._h(b + "attn.hook_pattern", scores.softmax(-1))
            z = self._h(b + "attn.hook_z", torch.einsum("bhqk,bkhe->bqhe", pattern, v))
            W_O = blk["W_O"].view(cfg.d_model, H, dh)
            if b + "attn.hook_result" in self._hooks:
                result = self._h(b + "attn.hook_result", torch.einsum("bqhe,dhe->bqhd", z, W_O))
                attn_out = result.sum(2)
            else:
                attn_out = z.reshape(B, P, H * dh) @ blk["W_O"].T
            attn_out = self._h(b + "hook_attn_out", attn_out)
            resid = self._h(b + "hook_resid_mid", resid + attn_out)
            x = self._h(b + "hook_mlp_in", self._rms(resid, blk["ln2"], b + "ln2.hook_scale"))
            post = self._h(b + "mlp.hook_post", F.silu(x @ blk["W_gate"].T) * (x @ blk["W_up"].T))
            mlp_out = self._h(b + "hook_mlp_out", post @ blk["W_down"].T)
            resid = self._h(b + "hook_resid_post", resid + mlp_out)
        if stop_at_layer is not None:
            return resid
        normed = self._rms(resid, self.ln_f[0], "ln_final.hook_scale")
        return self.unembed(normed[:, -1:] if last_only else normed)

    __call__ = forward

    def ln_final(self, resid: torch.Tensor) -> torch.Tensor:
        return resid / (resid.pow(2).mean(-1, keepdim=True) + self.cfg.ln_eps).sqrt() * self.ln_f[0]

    def to_tokens(self, text: str | list[str], prepend_bos: bool = False) -> torch.Tensor:
        return super().to_tokens(text, prepend_bos=prepend_bos)

    def to_single_token(self, s: str) -> int:
        ids = self.tokenizer.encode(s, prepend_bos=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} tokens")
        return ids[0]
