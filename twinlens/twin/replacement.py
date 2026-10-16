"""The digital twin: a transcoder-based replacement of GPT-2 that predicts intervention effects.

Several levels of model are run with the same hook machinery (DECISIONS.md, D3):

* ``"direct"``      — like "linear", but downstream features are held at their clean values, so
                      only the edited feature's *direct* path (residual + frozen attention) acts.
                      This is what reading a single edge of an attribution graph predicts.
* ``"linear"``      — attention patterns, LayerNorm scales and transcoder error terms are frozen
                      at the clean run, and each downstream feature's ReLU is linearised (its
                      clean on/off gate is kept). The model is then *affine* in any perturbation:
                      this is exactly the object an attribution graph describes.
* ``"replacement"`` — same frozen attention/LN/error, but features are recomputed with their
                      ReLU (the "local replacement model" of Ameisen et al., 2025).
* ``"replacement_ln"`` / ``"replacement_full"`` — the same with live LayerNorm scales, and then
                      also live attention patterns: only the transcoders (and their frozen error
                      terms) remain. Together these attribute the twin's error to its sources.
* ``"real"``        — the real GPT-2. A feature edit becomes the edit of its decoder direction,
                      ``mlp_out[pos] += (new - old) * W_dec[f]``, and everything is recomputed.

At zero perturbation every level gives the clean logits exactly (the error terms make the
replacement exact), so any difference under an intervention is the twin's prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from twinlens.core.gpt2 import GPT2
from twinlens.twin.transcoders import TranscoderSet, layernorm_pre

Mode = Literal["direct", "linear", "replacement", "replacement_ln", "replacement_full", "real"]

# The twin "ladder": each level unfreezes one more component of the real model.
#                      features      LN scales  attention
LEVELS: dict[str, tuple[str, bool, bool]] = {
    "direct": ("frozen", True, True),
    "linear": ("gated", True, True),
    "replacement": ("relu", True, True),
    "replacement_ln": ("relu", False, True),
    "replacement_full": ("relu", False, False),
}
TWIN_LEVELS: tuple[Mode, ...] = tuple(LEVELS)  # type: ignore[assignment]


@dataclass(frozen=True)
class FeatureEdit:
    layer: int
    pos: int
    feature: int
    value: float  # new activation (0.0 = ablate)


@dataclass
class CleanTrace:
    tokens: torch.Tensor  # [1, pos]
    logits: torch.Tensor  # [pos, vocab]
    patterns: dict[int, torch.Tensor]  # [1, head, q, k]
    scales: dict[str, torch.Tensor]  # LN scale hooks -> [1, pos, 1]
    resid_mid: dict[int, torch.Tensor]  # [1, pos, d_model]
    mlp_out: dict[int, torch.Tensor]  # real MLP outputs [1, pos, d_model]
    pre: dict[int, torch.Tensor]  # transcoder pre-activations [pos, d_sae]
    acts: dict[int, torch.Tensor]  # transcoder activations [pos, d_sae]
    error: dict[int, torch.Tensor]  # mlp_out - transcoder(mlp_in) [1, pos, d_model]
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def n_pos(self) -> int:
        return self.tokens.shape[1]

    def active(self, layer: int, pos: int) -> torch.Tensor:
        return self.acts[layer][pos].nonzero()[:, 0]


class Twin:
    def __init__(self, model: GPT2, transcoders: TranscoderSet | None = None):
        self.model = model
        self.tcs = transcoders or TranscoderSet(model.cfg.n_layers)
        self._labels: dict[tuple[int, int], list[str]] = {}

    # ------------------------------------------------------------------ tracing
    def trace(self, tokens: torch.Tensor) -> CleanTrace:
        m, L = self.model, self.model.cfg.n_layers
        if tokens.dim() == 1:
            tokens = tokens[None]
        names = ["ln_final.hook_scale"]
        for l in range(L):
            b = f"blocks.{l}."
            names += [
                b + "attn.hook_pattern",
                b + "ln1.hook_scale",
                b + "ln2.hook_scale",
                b + "hook_resid_mid",
                b + "hook_mlp_out",
            ]
        logits, cache = m.run_with_cache(tokens, names)
        pre, acts, error = {}, {}, {}
        for l in range(L):
            b = f"blocks.{l}."
            x = layernorm_pre(cache[b + "hook_resid_mid"], cache[b + "ln2.hook_scale"])
            tc = self.tcs[l]
            pre[l] = tc.pre(x)[0]
            acts[l] = torch.relu(pre[l])
            error[l] = cache[b + "hook_mlp_out"] - tc.decode(acts[l])[None]
        return CleanTrace(
            tokens=tokens,
            logits=logits[0],
            patterns={l: cache[f"blocks.{l}.attn.hook_pattern"] for l in range(L)},
            scales={k: v for k, v in cache.items() if k.endswith("hook_scale")},
            resid_mid={l: cache[f"blocks.{l}.hook_resid_mid"] for l in range(L)},
            mlp_out={l: cache[f"blocks.{l}.hook_mlp_out"] for l in range(L)},
            pre=pre,
            acts=acts,
            error=error,
        )

    # ------------------------------------------------------------- simulation
    def run(self, trace: CleanTrace, mode: Mode, edits: list[FeatureEdit | None]) -> torch.Tensor:
        """Run one batch row per edit (``None`` = no edit); return last-position logits."""
        m, L = self.model, self.model.cfg.n_layers
        B = len(edits)
        tokens = trace.tokens.expand(B, -1)
        real = [e for e in edits if e is not None]
        start = min((e.layer for e in real), default=L)
        by_layer: dict[int, list[tuple[int, FeatureEdit]]] = {}
        for row, e in enumerate(edits):
            if e is not None:
                by_layer.setdefault(e.layer, []).append((row, e))

        hooks = []
        if mode == "real":
            for l, items in by_layer.items():

                def edit_out(out: torch.Tensor, _: str, items=items, l=l) -> torch.Tensor:
                    out = out.clone()
                    W_dec = self.tcs[l].W_dec
                    for row, e in items:
                        old = trace.acts[l][e.pos, e.feature]
                        out[row, e.pos] += (e.value - old) * W_dec[e.feature]
                    return out

                hooks.append((f"blocks.{l}.hook_mlp_out", edit_out))
        else:
            features, freeze_ln, freeze_attn = LEVELS[mode]
            if freeze_ln:
                for name, s in trace.scales.items():
                    hooks.append((name, lambda x, _, s=s: s.expand_as(x)))
            if freeze_attn:
                for l in range(L):
                    pat = trace.patterns[l]
                    hooks.append(
                        (f"blocks.{l}.attn.hook_pattern", lambda x, _, p=pat: p.expand_as(x))
                    )
            store: dict[int, torch.Tensor] = {}
            for l in range(start, L):
                if features != "frozen":
                    hooks.append(
                        (f"blocks.{l}.hook_resid_mid", lambda x, _, l=l: store.__setitem__(l, x))
                    )
                hooks.append(
                    (
                        f"blocks.{l}.hook_mlp_out",
                        self._twin_mlp(trace, features, freeze_ln, l, store, by_layer.get(l, [])),
                    )
                )

        with m.hooks(hooks):
            return m(tokens, last_only=True)[:, -1]

    def _twin_mlp(self, trace: CleanTrace, features: str, freeze_ln: bool, l: int, store, items):
        """Replacement for ``hook_mlp_out`` at layer ``l``.

        Activations are sparse, so only the needed columns are computed:
        * frozen: downstream features keep their clean values -> clean output plus edit deltas;
        * gated: the ReLU gates are frozen -> only features active in the clean run matter;
        * relu: full encode (new features may turn on), then a sparse decode.
        """
        tc = self.tcs[l]
        scale = trace.scales[f"blocks.{l}.ln2.hook_scale"] if freeze_ln else None
        clean_acts = trace.acts[l]

        def edit_deltas(out: torch.Tensor) -> torch.Tensor:
            for row, e in items:
                out[row, e.pos] += (e.value - clean_acts[e.pos, e.feature]) * tc.W_dec[e.feature]
            return out

        if features == "frozen":
            return lambda out, _: edit_deltas(trace.mlp_out[l].expand_as(out).clone())

        if features == "gated":
            idx = torch.unique(
                torch.cat(
                    [
                        (trace.pre[l] > 0).any(0).nonzero()[:, 0],
                        torch.tensor([e.feature for _, e in items], dtype=torch.long),
                    ]
                )
            )
            W_enc, b_enc, W_dec = tc.W_enc[:, idx], tc.b_enc[idx], tc.W_dec[idx]
            gate = (trace.pre[l][:, idx] > 0).float()
            col = {int(f): i for i, f in enumerate(idx.tolist())}

            def linear_fn(out: torch.Tensor, _: str) -> torch.Tensor:
                acts = (layernorm_pre(store[l], scale) @ W_enc + b_enc) * gate
                for row, e in items:
                    acts[row, e.pos, col[e.feature]] = e.value
                return acts @ W_dec + tc.b_dec + trace.error[l]

            return linear_fn

        def replacement_fn(out: torch.Tensor, _: str) -> torch.Tensor:
            acts = tc.encode(layernorm_pre(store[l], scale))
            for row, e in items:
                acts[row, e.pos, e.feature] = e.value
            idx = acts.flatten(0, 1).any(0).nonzero()[:, 0]
            return acts[..., idx] @ tc.W_dec[idx] + tc.b_dec + trace.error[l]

        return replacement_fn

    # -------------------------------------------------------------- attribution
    def feature_label(self, layer: int, feature: int, k: int = 6) -> list[str]:
        """Offline label: the tokens whose logits the feature's decoder direction raises most
        (a logit-lens reading of ``W_dec[f]``)."""
        key = (layer, feature)
        if key not in self._labels:
            direction = self.tcs[layer].W_dec[feature] * self.model.ln_f[0]
            top = self.model.unembed(direction).topk(k).indices
            self._labels[key] = [self.model.tokenizer.token_str(int(i)) for i in top]
        return self._labels[key]

    def feature_effects(self, trace: CleanTrace, token: int) -> dict[int, torch.Tensor]:
        """Direct effect of every feature on ``logit[token]`` at the last position.

        Features are injected as constant leaves (stop-gradient) with attention and LN frozen,
        so gradient x activation is each feature's direct effect through the residual stream
        and attention, i.e. its edge weight to the logit node in an attribution graph.
        Returns ``{layer: [pos, d_sae]}``.
        """
        logit, leaves = self._leaf_forward(trace, lambda logits: logits[0, -1, token])
        grads = torch.autograd.grad(logit, list(leaves.values()))
        return {l: (g * leaf.detach()) for (l, leaf), g in zip(leaves.items(), grads, strict=True)}

    def attribute_logit(self, trace: CleanTrace, token: int, top: int = 20) -> list[dict]:
        """The ``top`` features with the largest |direct effect| on ``logit[token]``."""
        rows = []
        for l, eff in self.feature_effects(trace, token).items():
            idx = eff.flatten().abs().topk(min(top, eff.numel())).indices
            for i in idx.tolist():
                p, f = divmod(i, eff.shape[1])
                if trace.acts[l][p, f] > 0:
                    rows.append(
                        {
                            "layer": l,
                            "pos": p,
                            "feature": f,
                            "activation": trace.acts[l][p, f].item(),
                            "effect": eff[p, f].item(),
                        }
                    )
        rows.sort(key=lambda r: -abs(r["effect"]))
        return rows[:top]

    def _leaf_forward(self, trace: CleanTrace, target_fn):
        m, L = self.model, self.model.cfg.n_layers
        leaves = {l: trace.acts[l].clone().requires_grad_(True) for l in range(L)}
        hooks = [(n, lambda x, _, s=s: s.expand_as(x)) for n, s in trace.scales.items()]
        for l in range(L):
            hooks.append(
                (f"blocks.{l}.attn.hook_pattern", lambda x, _, p=trace.patterns[l]: p.expand_as(x))
            )
            hooks.append(
                (
                    f"blocks.{l}.hook_mlp_out",
                    lambda x, _, l=l: (self.tcs[l].decode(leaves[l]) + trace.error[l][0])[None],
                )
            )
        with torch.enable_grad(), m.hooks(hooks):
            logits = m(trace.tokens, last_only=True)
            return target_fn(logits), leaves
