"""A minimal, hookable GPT-2 in plain PyTorch.

Hook names follow TransformerLens conventions so results are directly comparable with the
interpretability literature:

    hook_embed, hook_pos_embed
    blocks.{l}.hook_resid_pre / hook_resid_mid / hook_resid_post
    blocks.{l}.ln1.hook_scale, blocks.{l}.ln2.hook_scale, ln_final.hook_scale
    blocks.{l}.attn.hook_q / hook_k / hook_v / hook_z        [batch, pos, head, d_head]
    blocks.{l}.attn.hook_pattern                              [batch, head, query, key]
    blocks.{l}.attn.hook_result                               [batch, pos, head, d_model]
    blocks.{l}.hook_attn_out, blocks.{l}.hook_mlp_in
    blocks.{l}.mlp.hook_pre / hook_post                       [batch, pos, d_mlp]
    blocks.{l}.hook_mlp_out

A hook is ``fn(tensor, name) -> tensor | None``; returning a tensor replaces the activation.
``hook_result`` (per-head outputs) is only materialised when something is attached to it.

Weights are memory-mapped from the Hugging Face safetensors file and never copied.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from twinlens.core.weights import MmapSafetensors, hf_file

HookFn = Callable[[torch.Tensor, str], torch.Tensor | None]


@dataclass(frozen=True)
class Config:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_mlp: int = 3072
    n_ctx: int = 1024
    d_vocab: int = 50257
    ln_eps: float = 1e-5

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


CONFIGS = {
    "gpt2": Config(),
    "gpt2-medium": Config(n_layers=24, n_heads=16, d_model=1024, d_mlp=4096),
}


class Tokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _Tok

        self._tok = _Tok.from_file(str(path))
        self.bos_id = self._tok.token_to_id("<|endoftext|>")

    def encode(self, text: str, prepend_bos: bool = True) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        return [self.bos_id, *ids] if prepend_bos else ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids))

    def token_str(self, i: int) -> str:
        return self._tok.decode([int(i)])

    def str_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.token_str(i) for i in ids]


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class GPT2:
    """Hookable GPT-2 forward pass over memory-mapped weights (inference only)."""

    def __init__(self, weights: MmapSafetensors, cfg: Config, tokenizer: Tokenizer):
        torch.set_grad_enabled(False)
        self.cfg = cfg
        self.tokenizer = tokenizer
        w = weights.get
        H, dh, D = cfg.n_heads, cfg.d_head, cfg.d_model
        self.W_E = w("wte.weight")  # [vocab, d_model]; tied unembedding
        self.W_pos = w("wpe.weight")
        self.ln_f = (w("ln_f.weight"), w("ln_f.bias"))
        self.blocks = []
        for layer in range(cfg.n_layers):
            p = f"h.{layer}."
            c_attn_w = w(p + "attn.c_attn.weight")  # [D, 3D]
            c_attn_b = w(p + "attn.c_attn.bias")
            self.blocks.append(
                {
                    "ln1": (w(p + "ln_1.weight"), w(p + "ln_1.bias")),
                    "ln2": (w(p + "ln_2.weight"), w(p + "ln_2.bias")),
                    # [D, 3, H, dh] view without copying
                    "W_qkv": c_attn_w.view(D, 3, H, dh),
                    "b_qkv": c_attn_b.view(3, H, dh),
                    "W_O": w(p + "attn.c_proj.weight").view(H, dh, D),
                    "b_O": w(p + "attn.c_proj.bias"),
                    "W_in": w(p + "mlp.c_fc.weight"),  # [D, d_mlp]
                    "b_in": w(p + "mlp.c_fc.bias"),
                    "W_out": w(p + "mlp.c_proj.weight"),  # [d_mlp, D]
                    "b_out": w(p + "mlp.c_proj.bias"),
                }
            )
        self._hooks: dict[str, list[HookFn]] = defaultdict(list)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, name: str = "gpt2") -> GPT2:
        repo = {"gpt2": "openai-community/gpt2", "gpt2-medium": "openai-community/gpt2-medium"}[
            name
        ]
        weights = MmapSafetensors(hf_file(repo, "model.safetensors"))
        return cls(weights, CONFIGS[name], Tokenizer(hf_file(repo, "tokenizer.json")))

    # -------------------------------------------------------------------- hooks
    def hook_names(self) -> list[str]:
        names = ["hook_embed", "hook_pos_embed", "ln_final.hook_scale"]
        for layer in range(self.cfg.n_layers):
            b = f"blocks.{layer}."
            names += [
                b + s
                for s in (
                    "hook_resid_pre",
                    "ln1.hook_scale",
                    "attn.hook_q",
                    "attn.hook_k",
                    "attn.hook_v",
                    "attn.hook_pattern",
                    "attn.hook_z",
                    "attn.hook_result",
                    "hook_attn_out",
                    "hook_resid_mid",
                    "ln2.hook_scale",
                    "hook_mlp_in",
                    "mlp.hook_pre",
                    "mlp.hook_post",
                    "hook_mlp_out",
                    "hook_resid_post",
                )
            ]
        return names

    @contextmanager
    def hooks(self, fwd_hooks: Iterable[tuple[str, HookFn]] | dict[str, HookFn]) -> Iterator[None]:
        items = list(fwd_hooks.items()) if isinstance(fwd_hooks, dict) else list(fwd_hooks)
        for name, fn in items:
            self._hooks[name].append(fn)
        try:
            yield
        finally:
            for name, fn in items:
                self._hooks[name].remove(fn)
                if not self._hooks[name]:
                    del self._hooks[name]

    def _h(self, name: str, x: torch.Tensor) -> torch.Tensor:
        for fn in self._hooks.get(name, ()):
            out = fn(x, name)
            if out is not None:
                x = out
        return x

    # ------------------------------------------------------------------ forward
    def _ln(
        self, x: torch.Tensor, wb: tuple[torch.Tensor, torch.Tensor], name: str
    ) -> torch.Tensor:
        x = x - x.mean(-1, keepdim=True)
        scale = self._h(name, (x.pow(2).mean(-1, keepdim=True) + self.cfg.ln_eps).sqrt())
        return x / scale * wb[0] + wb[1]

    def forward(
        self, tokens: torch.Tensor, stop_at_layer: int | None = None, last_only: bool = False
    ) -> torch.Tensor:
        """tokens [batch, pos] -> logits [batch, pos, vocab] (or resid if stop_at_layer).

        ``last_only`` unembeds only the final position ([batch, 1, vocab]), which saves a
        [batch, pos, 50257] allocation in batched patching experiments.
        """
        if tokens.dim() == 1:
            tokens = tokens[None]
        cfg = self.cfg
        n_pos = tokens.shape[1]
        resid = self._h("hook_embed", self.W_E[tokens]) + self._h(
            "hook_pos_embed", self.W_pos[:n_pos][None].expand(tokens.shape[0], -1, -1)
        )
        mask = torch.ones(n_pos, n_pos, dtype=torch.bool).tril()
        for layer, blk in enumerate(self.blocks[:stop_at_layer]):
            b = f"blocks.{layer}."
            resid = self._h(b + "hook_resid_pre", resid)
            x = self._ln(resid, blk["ln1"], b + "ln1.hook_scale")
            qkv = torch.einsum("bpd,dkhe->kbphe", x, blk["W_qkv"]) + blk["b_qkv"][:, None, None]
            q = self._h(b + "attn.hook_q", qkv[0])
            k = self._h(b + "attn.hook_k", qkv[1])
            v = self._h(b + "attn.hook_v", qkv[2])
            scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(cfg.d_head)
            scores = scores.masked_fill(~mask, float("-inf"))
            pattern = self._h(b + "attn.hook_pattern", scores.softmax(-1))
            z = self._h(b + "attn.hook_z", torch.einsum("bhqk,bkhe->bqhe", pattern, v))
            if b + "attn.hook_result" in self._hooks:
                result = self._h(
                    b + "attn.hook_result", torch.einsum("bqhe,hed->bqhd", z, blk["W_O"])
                )
                attn_out = result.sum(2) + blk["b_O"]
            else:
                attn_out = torch.einsum("bqhe,hed->bqd", z, blk["W_O"]) + blk["b_O"]
            attn_out = self._h(b + "hook_attn_out", attn_out)
            resid = self._h(b + "hook_resid_mid", resid + attn_out)
            x = self._h(b + "hook_mlp_in", self._ln(resid, blk["ln2"], b + "ln2.hook_scale"))
            pre = self._h(b + "mlp.hook_pre", x @ blk["W_in"] + blk["b_in"])
            post = self._h(b + "mlp.hook_post", _gelu_new(pre))
            mlp_out = self._h(b + "hook_mlp_out", post @ blk["W_out"] + blk["b_out"])
            resid = self._h(b + "hook_resid_post", resid + mlp_out)
        if stop_at_layer is not None:
            return resid
        normed = self._ln(resid, self.ln_f, "ln_final.hook_scale")
        return self.unembed(normed[:, -1:] if last_only else normed)

    __call__ = forward

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_E.T

    def ln_final(self, resid: torch.Tensor) -> torch.Tensor:
        """Final LayerNorm with fresh statistics (used by the logit lens)."""
        x = resid - resid.mean(-1, keepdim=True)
        x = x / (x.pow(2).mean(-1, keepdim=True) + self.cfg.ln_eps).sqrt()
        return x * self.ln_f[0] + self.ln_f[1]

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        names_filter: Callable[[str], bool] | Iterable[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if names_filter is None:
            names = [n for n in self.hook_names() if not n.endswith("hook_result")]
        elif callable(names_filter):
            names = [n for n in self.hook_names() if names_filter(n)]
        else:
            names = list(names_filter)
        cache: dict[str, torch.Tensor] = {}

        def save(x: torch.Tensor, name: str) -> None:
            cache[name] = x.detach()

        with self.hooks([(n, save) for n in names]):
            logits = self.forward(tokens)
        return logits, cache

    # --------------------------------------------------------------- utilities
    def to_tokens(self, text: str | list[str], prepend_bos: bool = True) -> torch.Tensor:
        texts = [text] if isinstance(text, str) else text
        ids = [self.tokenizer.encode(t, prepend_bos) for t in texts]
        if len({len(i) for i in ids}) != 1:
            raise ValueError("prompts in a batch must have the same number of tokens")
        return torch.tensor(ids)

    def to_single_token(self, s: str) -> int:
        ids = self.tokenizer.encode(s, prepend_bos=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} tokens")
        return ids[0]

    def generate(
        self, text: str, max_new_tokens: int = 20, temperature: float = 0.0
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield (token_id, logits_of_last_position) one step at a time."""
        tokens = self.to_tokens(text)
        for _ in range(max_new_tokens):
            logits = self.forward(tokens, last_only=True)[0, -1]
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                nxt = int(torch.multinomial(F.softmax(logits / temperature, -1), 1))
            yield nxt, logits
            tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)
            if tokens.shape[1] >= self.cfg.n_ctx:
                break
