"""A hookable Gemma-2 (four RMSNorms per block, GQA, RoPE, logit softcaps, GeGLU) in plain PyTorch.

Same hook names and interface as ``GPT2``/``Qwen3`` so the twin and the fidelity engine work
unchanged. Gemma-specific hooks (TransformerLens naming for the two extra norms):

    blocks.{l}.ln1       input_layernorm             before attention
    blocks.{l}.ln1_post  post_attention_layernorm    on the attention output (hook_attn_out: after)
    blocks.{l}.ln2       pre_feedforward_layernorm   hook_mlp_in is its output, weight included
    blocks.{l}.ln2_post  post_feedforward_layernorm  hook_mlp_out is after it
    blocks.{l}.mlp.hook_out   the MLP output before ln2_post

Every norm has ``.hook_scale`` and ``.hook_normalized`` (x / rms, before the ``1 + weight``
factor). ``attn.hook_q``/``hook_k`` are after RoPE, ``hook_k``/``hook_v`` are expanded from the KV
heads to the query heads (GQA). ``attn.hook_pattern`` is the softmax of the softcapped scores;
``attn.hook_result`` is per head, before ``ln1_post``.

The final logit softcap ``tanh(l / 30) * 30`` is applied at the end of ``forward`` (``unembed``
itself stays linear); ``final_softcap = None`` switches it off.

Weights are a memory-mapped F32 copy of the BF16 checkpoint (see ``weights.convert_to_f32``).
A BOS token is prepended, as Gemma expects.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from twinlens.core.gpt2 import Tokenizer
from twinlens.core.qwen3 import Qwen3
from twinlens.core.weights import CACHE_DIR, MmapSafetensors, convert_to_f32, hf_file

REPO = "unsloth/gemma-2-2b"  # ungated mirror of google/gemma-2-2b (same weights)


@dataclass(frozen=True)
class GemmaConfig:
    n_layers: int = 26
    n_heads: int = 8
    n_kv_heads: int = 4
    d_model: int = 2304
    d_head: int = 256
    d_mlp: int = 9216
    d_vocab: int = 256000
    n_ctx: int = 8192
    ln_eps: float = 1e-6
    rope_theta: float = 10_000.0
    query_pre_attn_scalar: float = 256.0
    attn_softcap: float | None = 50.0
    final_softcap: float | None = 30.0
    sliding_window: int = 4096
    # True for the layers using sliding-window attention (HF default: every even layer)
    sliding: tuple[bool, ...] | None = None
    bos_id: int = 2

    def is_sliding(self, layer: int) -> bool:
        return self.sliding[layer] if self.sliding is not None else layer % 2 == 0

    @classmethod
    def from_hf(cls, hf) -> GemmaConfig:
        """From a ``transformers.Gemma2Config`` or the dict in its ``config.json``."""
        get = hf.get if isinstance(hf, dict) else lambda k, d=None: getattr(hf, k, d)
        types = get("layer_types")
        kw = {
            "n_layers": get("num_hidden_layers"),
            "n_heads": get("num_attention_heads"),
            "n_kv_heads": get("num_key_value_heads"),
            "d_model": get("hidden_size"),
            "d_head": get("head_dim"),
            "d_mlp": get("intermediate_size"),
            "d_vocab": get("vocab_size"),
            "n_ctx": get("max_position_embeddings"),
            "ln_eps": get("rms_norm_eps"),
            # transformers >= 5 moves it into rope_parameters
            "rope_theta": get("rope_theta") or (get("rope_parameters") or {}).get("rope_theta")
            or 10_000.0,
            "query_pre_attn_scalar": get("query_pre_attn_scalar"),
            "attn_softcap": get("attn_logit_softcapping"),
            "final_softcap": get("final_logit_softcapping"),
            "sliding_window": get("sliding_window"),
            "sliding": tuple(t == "sliding_attention" for t in types) if types else None,
            "bos_id": get("bos_token_id") or 2,
        }
        act = get("hidden_activation") or get("hidden_act")
        if act not in (None, "gelu_pytorch_tanh"):
            raise ValueError(f"unsupported activation {act!r}")
        return cls(**kw)


class GemmaTokenizer(Tokenizer):
    def __init__(self, path: str | Path, bos_id: int = 2):
        super().__init__(path)
        self.bos_id = bos_id


class Gemma2(Qwen3):
    """Hookable Gemma-2 forward pass (inference only). Reuses Qwen3's RMSNorm hooks and RoPE."""

    def __init__(self, weights: MmapSafetensors, cfg: GemmaConfig, tokenizer: Tokenizer | None):
        torch.set_grad_enabled(False)
        self.cfg = cfg  # type: ignore[assignment]
        self.tokenizer = tokenizer  # type: ignore[assignment]
        self.final_softcap = cfg.final_softcap
        w = weights.get

        def norm(key: str) -> torch.Tensor:  # Gemma's RMSNorm multiplies by (1 + weight)
            return 1.0 + w(key).float()

        self.W_E = w("model.embed_tokens.weight")  # [vocab, d]; tied unembedding
        self.ln_f = (norm("model.norm.weight"), torch.zeros(cfg.d_model))
        self.blocks = []
        for layer in range(cfg.n_layers):
            p = f"model.layers.{layer}."
            self.blocks.append(
                {
                    "ln1": norm(p + "input_layernorm.weight"),
                    "ln1_post": norm(p + "post_attention_layernorm.weight"),
                    "ln2": norm(p + "pre_feedforward_layernorm.weight"),
                    "ln2_post": norm(p + "post_feedforward_layernorm.weight"),
                    "W_Q": w(p + "self_attn.q_proj.weight"),  # [H*dh, d]
                    "W_K": w(p + "self_attn.k_proj.weight"),  # [Hkv*dh, d]
                    "W_V": w(p + "self_attn.v_proj.weight"),
                    "W_O": w(p + "self_attn.o_proj.weight"),  # [d, H*dh]
                    "W_gate": w(p + "mlp.gate_proj.weight"),
                    "W_up": w(p + "mlp.up_proj.weight"),
                    "W_down": w(p + "mlp.down_proj.weight"),
                }
            )
        # as HF: float32 inverse frequencies from integer exponents
        exps = torch.arange(0, cfg.d_head, 2, dtype=torch.int64).float() / cfg.d_head
        self._inv_freq = 1.0 / (cfg.rope_theta**exps)
        self._embed_scale = torch.tensor(cfg.d_model**0.5, dtype=torch.float32)
        self._hooks = defaultdict(list)

    @classmethod
    def from_files(
        cls, weights: str | Path, cfg: GemmaConfig, tokenizer: Tokenizer | None = None
    ) -> Gemma2:
        """Build from a local safetensors file (F32 or BF16) with HF parameter names."""
        return cls(MmapSafetensors(weights), cfg, tokenizer)

    @classmethod
    def load(cls) -> Gemma2:  # type: ignore[override]
        src = hf_file(REPO, "model.safetensors")
        f32 = convert_to_f32(src, CACHE_DIR / "twinlens" / "gemma-2-2b.f32.safetensors")
        cfg = GemmaConfig.from_hf(json.loads(hf_file(REPO, "config.json").read_text()))
        tok = GemmaTokenizer(hf_file(REPO, "tokenizer.json"), cfg.bos_id)
        return cls(MmapSafetensors(f32), cfg, tok)

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
                    "attn.hook_q",
                    "attn.hook_k",
                    "attn.hook_v",
                    "attn.hook_pattern",
                    "attn.hook_z",
                    "attn.hook_result",
                    "ln1_post.hook_scale",
                    "ln1_post.hook_normalized",
                    "hook_attn_out",
                    "hook_resid_mid",
                    "ln2.hook_scale",
                    "ln2.hook_normalized",
                    "hook_mlp_in",
                    "mlp.hook_post",
                    "mlp.hook_out",
                    "ln2_post.hook_scale",
                    "ln2_post.hook_normalized",
                    "hook_mlp_out",
                    "hook_resid_post",
                )
            ]
        return names

    def _mask(self, layer: int, n_pos: int) -> torch.Tensor:
        i = torch.arange(n_pos)
        mask = i[None] <= i[:, None]  # [query, key]
        if self.cfg.is_sliding(layer):
            mask &= i[None] > i[:, None] - self.cfg.sliding_window
        return mask

    def forward(
        self, tokens: torch.Tensor, stop_at_layer: int | None = None, last_only: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens[None]
        cfg = self.cfg
        B, P = tokens.shape
        H, Hkv, dh = cfg.n_heads, cfg.n_kv_heads, cfg.d_head
        rep = H // Hkv
        resid = self._h("hook_embed", self.W_E[tokens].float() * self._embed_scale)
        for layer, blk in enumerate(self.blocks[:stop_at_layer]):
            b = f"blocks.{layer}."
            resid = self._h(b + "hook_resid_pre", resid)
            x = self._rms(resid, blk["ln1"], b + "ln1.hook_scale")
            q = (x @ blk["W_Q"].T).view(B, P, H, dh)
            k = (x @ blk["W_K"].T).view(B, P, Hkv, dh)
            v = (x @ blk["W_V"].T).view(B, P, Hkv, dh)
            q = self._h(b + "attn.hook_q", self._rope(q))
            k = self._h(b + "attn.hook_k", self._rope(k).repeat_interleave(rep, dim=2))
            v = self._h(b + "attn.hook_v", v.repeat_interleave(rep, dim=2))
            scores = torch.einsum("bqhe,bkhe->bhqk", q, k) * cfg.query_pre_attn_scalar**-0.5
            if cfg.attn_softcap is not None:
                scores = torch.tanh(scores / cfg.attn_softcap) * cfg.attn_softcap
            scores = scores.masked_fill(~self._mask(layer, P), float("-inf"))
            pattern = self._h(b + "attn.hook_pattern", scores.softmax(-1))
            z = self._h(b + "attn.hook_z", torch.einsum("bhqk,bkhe->bqhe", pattern, v))
            W_O = blk["W_O"].view(cfg.d_model, H, dh)
            if b + "attn.hook_result" in self._hooks:
                result = self._h(b + "attn.hook_result", torch.einsum("bqhe,dhe->bqhd", z, W_O))
                attn = result.sum(2)
            else:
                attn = z.reshape(B, P, H * dh) @ blk["W_O"].T
            attn_out = self._h(
                b + "hook_attn_out", self._rms(attn, blk["ln1_post"], b + "ln1_post.hook_scale")
            )
            resid = self._h(b + "hook_resid_mid", resid + attn_out)
            x = self._h(b + "hook_mlp_in", self._rms(resid, blk["ln2"], b + "ln2.hook_scale"))
            gate = F.gelu(x @ blk["W_gate"].T, approximate="tanh")
            post = self._h(b + "mlp.hook_post", gate * (x @ blk["W_up"].T))
            out = self._h(b + "mlp.hook_out", post @ blk["W_down"].T)
            mlp_out = self._h(
                b + "hook_mlp_out", self._rms(out, blk["ln2_post"], b + "ln2_post.hook_scale")
            )
            resid = self._h(b + "hook_resid_post", resid + mlp_out)
        if stop_at_layer is not None:
            return resid
        normed = self._rms(resid, self.ln_f[0], "ln_final.hook_scale")
        logits = self.unembed(normed[:, -1:] if last_only else normed)
        if self.final_softcap is not None:
            logits = torch.tanh(logits / self.final_softcap) * self.final_softcap
        return logits

    __call__ = forward

    def to_tokens(self, text: str | list[str], prepend_bos: bool = True) -> torch.Tensor:
        return super(Qwen3, self).to_tokens(text, prepend_bos=prepend_bos)
