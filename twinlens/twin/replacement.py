"""The digital twin: a transcoder-based replacement model that predicts intervention effects.

Works for any model exposing the TwinLens hook interface (GPT-2, Qwen3). Several levels of model
are run with the same hook machinery (DECISIONS.md, D3):

* ``"direct"``      — like "linear", but downstream features are held at their clean values, so
                      only the edited feature's *direct* path (residual + frozen attention) acts.
                      This is what reading a single edge of an attribution graph predicts.
* ``"linear"``      — attention patterns, normalisation scales and transcoder error terms are
                      frozen at the clean run, and each downstream feature's ReLU is linearised (its
                      clean on/off gate is kept). The model is then *affine* in any perturbation:
                      this is exactly the object an attribution graph describes.
* ``"replacement"`` — same frozen attention/norms/error, but features are recomputed with their
                      ReLU (the "local replacement model" of Ameisen et al., 2025).
* ``"replacement_ln"`` / ``"replacement_full"`` — the same with live normalisation scales, and then
                      also live attention patterns: only the transcoders (and their frozen error
                      terms) remain. Together these attribute the twin's error to its sources.
* ``"linear_lnjac"`` / ``"linear_fulljac"`` — the proposed fix: the linear twin keeping the
                      first-order terms that freezing drops (normalisation scale, then also the
                      attention pattern). Still exactly linear.
* ``"real"``        — the real model. A feature edit becomes the edit of its decoder direction,
                      ``mlp_out[pos] += (new - old) * W_dec[f]``, and everything is recomputed.

At zero perturbation every level gives the clean logits exactly (the error terms make the
replacement exact), so any difference under an intervention is the twin's prediction error.

All transcoder compute is sparse (only the features active in the clean run, plus edited
ones), which is what makes 164k-feature transcoders usable on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from twinlens.core.gpt2 import GPT2
from twinlens.twin.transcoders import TranscoderSet

Mode = Literal[
    "direct",
    "linear",
    "linear_lnjac",
    "linear_lnlive",
    "linear_fulljac",
    "linear_live",
    "replacement",
    "replacement_ln",
    "replacement_full",
    "real",
]

# The twin "ladder": each level unfreezes one more component of the real model.
# Normalisation scales and attention patterns are each "frozen" (clean values, as in attribution
# graphs), "jacobian" (clean values plus their exact first-order change: the twin stays linear),
# or "live" (recomputed).
#                      features   norms        attention
LEVELS: dict[str, tuple[str, str, str]] = {
    "direct": ("frozen", "frozen", "frozen"),
    "linear": ("gated", "frozen", "frozen"),
    "replacement": ("relu", "frozen", "frozen"),
    "replacement_ln": ("relu", "live", "frozen"),
    "replacement_full": ("relu", "live", "live"),
    # Proposed fixes: the graph twin, keeping the derivatives that attribution graphs drop.
    "linear_lnjac": ("gated", "jacobian", "frozen"),
    "linear_fulljac": ("gated", "jacobian", "jacobian"),
    # References for testing the Jacobians (not linear).
    "linear_lnlive": ("gated", "live", "frozen"),
    "linear_live": ("gated", "live", "live"),
}
TWIN_LEVELS: tuple[Mode, ...] = (
    "direct",
    "linear",
    "replacement",
    "replacement_ln",
    "replacement_full",
)
FIX_LEVELS: tuple[Mode, ...] = ("linear_lnjac", "linear_fulljac")


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
    qkv: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # clean q, k, v [1,pos,h,dh]
    scales: dict[str, torch.Tensor]  # normalisation scale hooks -> [..., 1]
    normalized: dict[str, torch.Tensor]  # normalised-output hooks (before the norm weight)
    resid_mid: dict[int, torch.Tensor]  # [1, pos, d_model]
    mlp_out: dict[int, torch.Tensor]  # real MLP outputs [1, pos, d_model]
    acts: dict[int, torch.Tensor]  # transcoder activations [pos, d_sae] (twin layers only)
    error: dict[int, torch.Tensor]  # mlp_out - transcoder(mlp_in) [1, pos, d_model]
    active: dict[int, torch.Tensor] = field(default_factory=dict)  # feature ids active anywhere
    dec: dict[int, torch.Tensor] = field(default_factory=dict)  # their decoder rows [k, d_model]

    @property
    def n_pos(self) -> int:
        return self.tokens.shape[1]

    @property
    def pre(self) -> dict[int, torch.Tensor]:  # back-compat: only the sign is ever used
        return self.acts


class Twin:
    def __init__(
        self,
        model: GPT2,
        transcoders=None,
        layers=None,
        tc_hook: str | None = None,
        skip_first: bool | None = None,
    ):
        self.model = model
        self.tcs = transcoders or TranscoderSet(model.cfg.n_layers)
        self.layers: list[int] = list(
            layers if layers is not None else getattr(self.tcs, "layers", range(model.cfg.n_layers))
        )
        # Where the transcoder reads: GPT-2 transcoders read LayerNormPre(resid_mid) (no affine),
        # the Qwen3 ones read the RMSNorm output including its weight.
        self.tc_hook = tc_hook or (
            "hook_mlp_in" if hasattr(self.tcs, "layers") else "ln2.hook_normalized"
        )
        # Position 0 can never be changed by an edit (causal attention), so its MLP output is
        # constant and may live entirely in the error term. Qwen's first token is an attention
        # sink with a huge norm on which transcoders fire thousands of features; skipping it
        # is exact and keeps the twin sparse.
        self.skip_first = skip_first if skip_first is not None else hasattr(self.tcs, "layers")
        self._labels: dict[tuple[int, int], list[str]] = {}

    # ------------------------------------------------------------------ tracing
    def trace(self, tokens: torch.Tensor) -> CleanTrace:
        m, L = self.model, self.model.cfg.n_layers
        if tokens.dim() == 1:
            tokens = tokens[None]
        names = [n for n in m.hook_names() if n.endswith(("hook_scale", "hook_normalized"))]
        for l in range(L):
            b = f"blocks.{l}."
            names += [
                b + s
                for s in (
                    "attn.hook_pattern",
                    "attn.hook_q",
                    "attn.hook_k",
                    "attn.hook_v",
                    "hook_resid_mid",
                    "hook_mlp_out",
                    self.tc_hook,
                )
            ]
        logits, cache = m.run_with_cache(tokens, list(dict.fromkeys(names)))
        acts, error, active, dec = {}, {}, {}, {}
        for l in self.layers:
            b = f"blocks.{l}."
            tc = self.tcs[l]
            acts[l] = torch.relu(tc.pre(cache[b + self.tc_hook])[0])
            if self.skip_first:
                acts[l][0] = 0
            active[l] = acts[l].any(0).nonzero()[:, 0]
            dec[l] = tc.dec_rows(active[l])
            recon = acts[l][:, active[l]] @ dec[l] + tc.b_dec
            error[l] = cache[b + "hook_mlp_out"] - recon[None]
        return CleanTrace(
            tokens=tokens,
            logits=logits[0],
            patterns={l: cache[f"blocks.{l}.attn.hook_pattern"] for l in range(L)},
            qkv={l: tuple(cache[f"blocks.{l}.attn.hook_{x}"] for x in "qkv") for l in range(L)},
            scales={k: v for k, v in cache.items() if k.endswith("hook_scale")},
            normalized={k: v for k, v in cache.items() if k.endswith("hook_normalized")},
            resid_mid={l: cache[f"blocks.{l}.hook_resid_mid"] for l in range(L)},
            mlp_out={l: cache[f"blocks.{l}.hook_mlp_out"] for l in range(L)},
            acts=acts,
            error=error,
            active=active,
            dec=dec,
        )

    def dec_row(self, trace: CleanTrace, layer: int, feature: int) -> torch.Tensor:
        hit = (trace.active[layer] == feature).nonzero()
        if len(hit):
            return trace.dec[layer][int(hit[0, 0])]
        return self.tcs[layer].dec_rows([feature])[0]

    # ------------------------------------------------------------- simulation
    def run(self, trace: CleanTrace, mode: Mode, edits: list[FeatureEdit | None]) -> torch.Tensor:
        """Run one batch row per edit (``None`` = no edit); return last-position logits."""
        m = self.model
        B = len(edits)
        tokens = trace.tokens.expand(B, -1)
        real = [e for e in edits if e is not None]
        for e in real:
            if e.layer not in self.layers:
                raise ValueError(f"layer {e.layer} has no transcoder in this twin")
        start = min((e.layer for e in real), default=m.cfg.n_layers)
        by_layer: dict[int, list[tuple[int, FeatureEdit]]] = {}
        for row, e in enumerate(edits):
            if e is not None:
                by_layer.setdefault(e.layer, []).append((row, e))

        hooks = []
        if mode == "real":
            for l, items in by_layer.items():
                rows = {e.feature: self.dec_row(trace, l, e.feature) for _, e in items}

                def edit_out(
                    out: torch.Tensor, _: str, items=items, l=l, rows=rows
                ) -> torch.Tensor:
                    out = out.clone()
                    for row, e in items:
                        out[row, e.pos] += (e.value - trace.acts[l][e.pos, e.feature]) * rows[
                            e.feature
                        ]
                    return out

                hooks.append((f"blocks.{l}.hook_mlp_out", edit_out))
        else:
            features, ln_mode, attn_mode = LEVELS[mode]
            if ln_mode != "live":
                for name, s in trace.scales.items():
                    hooks.append((name, lambda x, _, s=s: s.expand_as(x)))
            if ln_mode == "jacobian":
                # n(u) = u / sigma(u) linearised at the clean point n0 = u0 / sigma0:
                #   dn = da - n0 (n0 . da) / d,  with a = u / sigma0 (frozen-scale output).
                # Same for LayerNorm (u centred) and RMSNorm (u = x); per head for QK-norm.
                for name, n0 in trace.normalized.items():

                    def ln_jac(a: torch.Tensor, _: str, n0=n0) -> torch.Tensor:
                        da = a - n0
                        return a - n0 * (da * n0).sum(-1, keepdim=True) / n0.shape[-1]

                    hooks.append((name, ln_jac))
            if attn_mode == "frozen":
                for l in range(m.cfg.n_layers):
                    pat = trace.patterns[l]
                    hooks.append(
                        (f"blocks.{l}.attn.hook_pattern", lambda x, _, p=pat: p.expand_as(x))
                    )
            elif attn_mode == "jacobian":
                hooks += self._attn_jacobian_hooks(trace)
            store: dict[int, torch.Tensor] = {}
            for l in range(start, m.cfg.n_layers):
                if l not in self.layers:
                    continue  # no transcoder: the real MLP stays (only above the first twin layer)
                if features != "frozen":
                    # The transcoder input hook already reflects the level's norm treatment.
                    hooks.append(
                        (f"blocks.{l}.{self.tc_hook}", lambda x, _, l=l: store.__setitem__(l, x))
                    )
                hooks.append(
                    (
                        f"blocks.{l}.hook_mlp_out",
                        self._twin_mlp(trace, features, l, store, by_layer.get(l, [])),
                    )
                )

        with m.hooks(hooks):
            return m(tokens, last_only=True)[:, -1]

    def _attn_jacobian_hooks(self, trace: CleanTrace) -> list:
        """First-order expansion of attention around the clean run:
        ds = (dq . k0 + q0 . dk) / sqrt(d_head),   dp = p0 * (ds - sum_k p0 ds),
        z  = p0 v + dp v0      (the second-order term dp dv is removed)."""
        hooks: list = []
        scale = self.model.cfg.d_head**0.5
        for l in range(self.model.cfg.n_layers):
            q0, k0, v0 = trace.qkv[l]
            p0 = trace.patterns[l]
            cur: dict[str, torch.Tensor] = {}
            for x in "qkv":
                hooks.append(
                    (f"blocks.{l}.attn.hook_{x}", lambda t, _, c=cur, x=x: c.__setitem__(x, t))
                )

            def pattern(_p: torch.Tensor, _: str, c=cur, q0=q0, k0=k0, p0=p0) -> torch.Tensor:
                dq, dk = c["q"] - q0, c["k"] - k0
                ds = (
                    torch.einsum("bqhe,bkhe->bhqk", dq, k0)
                    + torch.einsum("bqhe,bkhe->bhqk", q0, dk)
                ) / scale
                c["dp"] = p0 * (ds - (p0 * ds).sum(-1, keepdim=True))
                return p0 + c["dp"]

            def z(zt: torch.Tensor, _: str, c=cur, v0=v0) -> torch.Tensor:
                return zt - torch.einsum("bhqk,bkhe->bqhe", c["dp"], c["v"] - v0)

            hooks.append((f"blocks.{l}.attn.hook_pattern", pattern))
            hooks.append((f"blocks.{l}.attn.hook_z", z))
        return hooks

    def _twin_mlp(self, trace: CleanTrace, features: str, l: int, store, items):
        """Replacement for ``hook_mlp_out`` at layer ``l``, computed sparsely:
        * frozen: downstream features keep their clean values -> clean output plus edit deltas;
        * gated: the ReLU gates are frozen -> only features active in the clean run matter;
        * relu: full encode (new features may turn on), then a sparse decode.
        """
        tc = self.tcs[l]
        clean_acts = trace.acts[l]
        edit_rows = {e.feature: self.dec_row(trace, l, e.feature) for _, e in items}

        def edit_deltas(out: torch.Tensor) -> torch.Tensor:
            for row, e in items:
                out[row, e.pos] += (e.value - clean_acts[e.pos, e.feature]) * edit_rows[e.feature]
            return out

        if features == "frozen":
            return lambda out, _: edit_deltas(trace.mlp_out[l].expand_as(out).clone())

        if features == "gated":
            extra = [e.feature for _, e in items if e.feature not in set(trace.active[l].tolist())]
            idx = torch.cat([trace.active[l], torch.tensor(extra, dtype=torch.long)])
            W_enc, b_enc = tc.enc_rows(idx)  # [k, d_model], [k]
            W_dec = torch.cat([trace.dec[l], tc.dec_rows(extra)]) if extra else trace.dec[l]
            gate = (clean_acts[:, idx] > 0).float()
            col = {int(f): i for i, f in enumerate(idx.tolist())}

            def linear_fn(out: torch.Tensor, _: str) -> torch.Tensor:
                acts = (store[l] @ W_enc.T + b_enc) * gate
                for row, e in items:
                    acts[row, e.pos, col[e.feature]] = e.value
                return acts @ W_dec + tc.b_dec + trace.error[l]

            return linear_fn

        def replacement_fn(out: torch.Tensor, _: str) -> torch.Tensor:
            acts = torch.relu(tc.pre(store[l]))
            for row, e in items:
                acts[row, e.pos, e.feature] = e.value
            idx = acts.flatten(0, 1).any(0).nonzero()[:, 0]
            return acts[..., idx] @ tc.dec_rows(idx) + tc.b_dec + trace.error[l]

        return replacement_fn

    # -------------------------------------------------------------- attribution
    def feature_label(self, layer: int, feature: int, k: int = 6) -> list[str]:
        """Offline label: the tokens whose logits the feature's decoder direction raises most
        (a logit-lens reading of ``W_dec[f]``)."""
        key = (layer, feature)
        if key not in self._labels:
            direction = self.tcs[layer].dec_rows([feature])[0] * self.model.ln_f[0]
            top = self.model.unembed(direction).topk(k).indices
            self._labels[key] = [self.model.tokenizer.token_str(int(i)) for i in top]
        return self._labels[key]

    def feature_effects(self, trace: CleanTrace, token: int) -> dict[int, torch.Tensor]:
        """Direct effect of every feature on ``logit[token]`` at the last position.

        Features are injected as constant leaves (stop-gradient) with attention and norms frozen,
        so gradient x activation is each feature's direct effect through the residual stream
        and attention, i.e. its edge weight to the logit node in an attribution graph.
        Returns ``{layer: [pos, d_sae]}`` (zero for inactive features).
        """
        logit, leaves = self._leaf_forward(trace, lambda logits: logits[0, -1, token])
        grads = torch.autograd.grad(logit, list(leaves.values()))
        out = {}
        for (l, leaf), g in zip(leaves.items(), grads, strict=True):
            full = torch.zeros_like(trace.acts[l])
            full[:, trace.active[l]] = g * leaf.detach()
            out[l] = full
        return out

    def attribute_logit(
        self, trace: CleanTrace, token: int, top: int = 20, per_cell: int | None = None
    ) -> list[dict]:
        """The ``top`` features with the largest |direct effect| on ``logit[token]``.

        ``per_cell`` caps how many features one (layer, position) cell may contribute, so a
        display shows the circuit across tokens instead of one crowded cell.
        """
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
        if per_cell is not None:
            seen: dict[tuple[int, int], int] = {}
            kept = []
            for r in rows:
                cell = (r["layer"], r["pos"])
                if seen.get(cell, 0) < per_cell:
                    seen[cell] = seen.get(cell, 0) + 1
                    kept.append(r)
            rows = kept
        return rows[:top]

    def _leaf_forward(self, trace: CleanTrace, target_fn):
        m = self.model
        leaves = {
            l: trace.acts[l][:, trace.active[l]].clone().requires_grad_(True) for l in self.layers
        }
        hooks = [(n, lambda x, _, s=s: s.expand_as(x)) for n, s in trace.scales.items()]
        for l in range(m.cfg.n_layers):
            hooks.append(
                (f"blocks.{l}.attn.hook_pattern", lambda x, _, p=trace.patterns[l]: p.expand_as(x))
            )
        for l in self.layers:
            hooks.append(
                (
                    f"blocks.{l}.hook_mlp_out",
                    lambda x, _, l=l: (
                        leaves[l] @ trace.dec[l] + self.tcs[l].b_dec + trace.error[l][0]
                    )[None],
                )
            )
        with torch.enable_grad(), m.hooks(hooks):
            logits = m(trace.tokens, last_only=True)
            return target_fn(logits), leaves
