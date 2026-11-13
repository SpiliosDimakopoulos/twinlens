"""The digital twin: a transcoder-based replacement model that predicts intervention effects.

Works for any model exposing the TwinLens hook interface (GPT-2, Qwen3, Gemma-2). Several levels
of model are run with the same hook machinery (DECISIONS.md, D3):

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
* ``"real_frozen_attn"`` — the same with attention patterns frozen at the clean run, which is
                      circuit-tracer's default ``feature_intervention`` (DECISIONS.md, D5).

Per-layer transcoders (PLT) and cross-layer transcoders (CLT) share one code path: a feature of
source layer ``i`` writes one decoder vector into ``mlp_out[l]`` for each of its output layers
(PLT: ``l = i`` only; CLT: every ``l >= i``), so the twin's output at layer ``l`` is the sum over
all sources writing there, plus that layer's bias and error term.

At zero perturbation every level gives the clean logits exactly (the error terms make the
replacement exact), so any difference under an intervention is the twin's prediction error.

Gemma-2's final logit softcap ``30 tanh(l / 30)`` is treated as part of the read-out, like the
log-softmax of the metrics: the model applies it to the final logits at every level alike, so
"linear" levels are linear up to it (they are affine in the logits before the cap).

All transcoder compute is sparse (only the features active in the clean run, plus edited
ones), which is what makes 164k-feature transcoders usable on a CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import torch

from twinlens.core.gpt2 import GPT2, HookFn
from twinlens.twin.transcoders import GemmaScopeTranscoderSet, QwenTranscoderSet, TranscoderSet

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
    "real_frozen_attn",
]
REAL_MODES: tuple[Mode, ...] = ("real", "real_frozen_attn")

Treatment = Literal["frozen", "jacobian", "live"]


class Level(NamedTuple):
    features: Literal["frozen", "gated", "relu"]
    norms: Treatment
    attention: Treatment


# The twin "ladder": each level unfreezes one more component of the real model.
# Normalisation scales and attention patterns are each "frozen" (clean values, as in attribution
# graphs), "jacobian" (clean values plus their exact first-order change: the twin stays linear),
# or "live" (recomputed).
LEVELS: dict[Mode, Level] = {
    "direct": Level("frozen", "frozen", "frozen"),
    "linear": Level("gated", "frozen", "frozen"),
    "replacement": Level("relu", "frozen", "frozen"),
    "replacement_ln": Level("relu", "live", "frozen"),
    "replacement_full": Level("relu", "live", "live"),
    # Proposed fixes: the graph twin, keeping the derivatives that attribution graphs drop.
    "linear_lnjac": Level("gated", "jacobian", "frozen"),
    "linear_fulljac": Level("gated", "jacobian", "jacobian"),
    # References for testing the Jacobians (not linear).
    "linear_lnlive": Level("gated", "live", "frozen"),
    "linear_live": Level("gated", "live", "live"),
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
    acts: dict[int, torch.Tensor]  # transcoder activations [pos, d_sae] (source layers only)
    error: dict[int, torch.Tensor]  # mlp_out - transcoder output [1, pos, d_model] (output layers)
    active: dict[int, torch.Tensor] = field(default_factory=dict)  # feature ids active anywhere
    # their decoder rows [k, n_out, d_model]; slot j writes to the source's j-th output layer
    dec: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def n_pos(self) -> int:
        return self.tokens.shape[1]


def _freeze(clean: torch.Tensor) -> HookFn:
    """Hook replacing an activation by its clean value (broadcast over the batch)."""
    return lambda x, _: clean.expand_as(x)


def _record(store: dict, key) -> HookFn:
    """Hook saving the activation in ``store[key]`` without changing it."""
    return lambda x, _: store.__setitem__(key, x)


class Twin:
    def __init__(
        self,
        model: GPT2,
        transcoders: TranscoderSet | QwenTranscoderSet | GemmaScopeTranscoderSet | None = None,
        layers: Iterable[int] | None = None,
        tc_hook: str | None = None,
        skip_first: bool | None = None,
        tc_out_hook: str | None = None,
    ):
        self.model = model
        self.tcs = transcoders or TranscoderSet(model.cfg.n_layers)
        # The GPT-2 set has no ``layers`` list; the others may cover a subset of layers.
        qwen = hasattr(self.tcs, "layers")
        if layers is None:
            layers = self.tcs.layers if qwen else range(model.cfg.n_layers)
        self.layers: list[int] = list(layers)
        # Where the transcoder reads: GPT-2 transcoders read LayerNormPre(resid_mid) (no affine),
        # the Qwen3 ones read the RMSNorm output including its weight; the Gemma set says.
        self.tc_hook = (
            tc_hook
            or getattr(self.tcs, "in_hook", None)
            or ("hook_mlp_in" if qwen else "ln2.hook_normalized")
        )
        # Where it writes (and where edits and error terms live). For Gemma-2 this may be before
        # the post-feedforward norm, which then stays in the twin like any other norm.
        self.tc_out_hook = tc_out_hook or getattr(self.tcs, "out_hook", "hook_mlp_out")
        # Position 0 can never be changed by an edit (causal attention), so its MLP output is
        # constant and may live entirely in the error term. Qwen's first token is an attention
        # sink with a huge norm on which transcoders fire thousands of features; skipping it
        # is exact and keeps the twin sparse.
        if skip_first is None:
            skip_first = getattr(self.tcs, "skip_first", qwen)
        self.skip_first = skip_first
        # Cross-layer transcoders: a source layer writes to itself and every later layer.
        self.cross = bool(getattr(self.tcs, "cross_layer", False))
        L = model.cfg.n_layers
        self.sources: dict[int, list[int]] = {}  # output layer -> source layers writing there
        for i in self.layers:
            for l in self._outs(i):
                self.sources.setdefault(l, []).append(i)
        self.out_layers: list[int] = sorted(self.sources)
        if self.cross and any(l >= L for l in self.out_layers):
            raise ValueError("cross-layer transcoder writes past the last layer")
        self._labels: dict[tuple[int, int], list[str]] = {}

    def _outs(self, i: int) -> range:
        """Output layers of source layer ``i``."""
        return range(i, self.model.cfg.n_layers) if self.cross else range(i, i + 1)

    def _dec_rows(self, i: int, idx) -> torch.Tensor:
        """Decoder rows of source ``i`` as ``[k, n_out, d_model]`` (PLT rows get ``n_out = 1``)."""
        rows = self.tcs[i].dec_rows(idx)
        return rows if rows.dim() == 3 else rows[:, None]

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
                    self.tc_out_hook,
                    self.tc_hook,
                )
            ]
        logits, cache = m.run_with_cache(tokens, list(dict.fromkeys(names)))
        acts, error, active, dec = {}, {}, {}, {}
        for i in self.layers:
            acts[i] = self.tcs[i].encode(cache[f"blocks.{i}.{self.tc_hook}"])[0]  # (Jump)ReLU
            if self.skip_first:
                acts[i][0] = 0
            active[i] = acts[i].any(0).nonzero()[:, 0]
            dec[i] = self._dec_rows(i, active[i])
        for l in self.out_layers:
            recon = self._sum_sources(
                l, self.sources[l], lambda i, j: acts[i][:, active[i]] @ dec[i][:, j]
            )
            recon = recon + self.tcs[l].b_dec
            error[l] = cache[f"blocks.{l}.{self.tc_out_hook}"] - recon[None]
        return CleanTrace(
            tokens=tokens,
            logits=logits[0],
            patterns={l: cache[f"blocks.{l}.attn.hook_pattern"] for l in range(L)},
            qkv={l: tuple(cache[f"blocks.{l}.attn.hook_{x}"] for x in "qkv") for l in range(L)},
            scales={k: v for k, v in cache.items() if k.endswith("hook_scale")},
            normalized={k: v for k, v in cache.items() if k.endswith("hook_normalized")},
            resid_mid={l: cache[f"blocks.{l}.hook_resid_mid"] for l in range(L)},
            mlp_out={l: cache[f"blocks.{l}.{self.tc_out_hook}"] for l in range(L)},
            acts=acts,
            error=error,
            active=active,
            dec=dec,
        )

    @staticmethod
    def _sum_sources(l: int, sources: list[int], term) -> torch.Tensor | None:
        """``sum_i term(i, l - i)`` over ``sources`` in ascending order (``None`` if empty).
        With one source (PLT) this is exactly that term, so old results are unchanged."""
        total = None
        for i in sources:
            t = term(i, l - i)
            total = t if total is None else total + t
        return total

    def dec_row(self, trace: CleanTrace, layer: int, feature: int) -> torch.Tensor:
        """Decoder vectors ``[n_out, d_model]`` of one feature (``n_out = 1`` for a PLT)."""
        hit = (trace.active[layer] == feature).nonzero()
        if len(hit):
            return trace.dec[layer][int(hit[0, 0])]
        return self._dec_rows(layer, [feature])[0]

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

        if mode in REAL_MODES:
            hooks = self._real_hooks(trace, by_layer)
            if mode == "real_frozen_attn":
                hooks += [
                    (f"blocks.{l}.attn.hook_pattern", _freeze(trace.patterns[l]))
                    for l in range(m.cfg.n_layers)
                ]
        else:
            hooks = self._twin_hooks(trace, LEVELS[mode], start, by_layer)
        with m.hooks(hooks):
            return m(tokens, last_only=True)[:, -1]

    def _edit_deltas(
        self, trace: CleanTrace, l: int, by_layer: dict[int, list[tuple[int, FeatureEdit]]]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """In place: ``out[row, pos] += (new - clean) * W_dec[f -> l]`` for each edit whose
        feature writes to output layer ``l``. Every edit sets its feature relative to the clean
        activation; its own input is upstream of the edit, so that is also its current one."""
        terms = []
        for i, items in by_layer.items():
            if l not in self._outs(i):
                continue
            rows = {e.feature: self.dec_row(trace, i, e.feature)[l - i] for _, e in items}
            terms += [(row, e, trace.acts[i], rows[e.feature]) for row, e in items]

        def add(out: torch.Tensor) -> torch.Tensor:
            for row, e, clean_acts, vec in terms:
                out[row, e.pos] += (e.value - clean_acts[e.pos, e.feature]) * vec
            return out

        return add

    def _real_hooks(
        self, trace: CleanTrace, by_layer: dict[int, list[tuple[int, FeatureEdit]]]
    ) -> list[tuple[str, HookFn]]:
        def edit_out(add: Callable[[torch.Tensor], torch.Tensor]) -> HookFn:
            return lambda out, _: add(out.clone())

        outs = sorted({l for i in by_layer for l in self._outs(i)})
        return [
            (f"blocks.{l}.{self.tc_out_hook}", edit_out(self._edit_deltas(trace, l, by_layer)))
            for l in outs
        ]

    def _twin_hooks(
        self,
        trace: CleanTrace,
        level: Level,
        start: int,
        by_layer: dict[int, list[tuple[int, FeatureEdit]]],
    ) -> list[tuple[str, HookFn]]:
        m = self.model
        hooks: list[tuple[str, HookFn]] = []
        if level.norms != "live":
            for name, s in trace.scales.items():
                hooks.append((name, _freeze(s)))
        if level.norms == "jacobian":
            # n(u) = u / sigma(u) linearised at the clean point n0 = u0 / sigma0:
            #   dn = da - n0 (n0 . da) / d,  with a = u / sigma0 (frozen-scale output).
            # Same for LayerNorm (u centred) and RMSNorm (u = x); per head for QK-norm.
            for name, n0 in trace.normalized.items():

                def ln_jac(a: torch.Tensor, _: str, n0=n0) -> torch.Tensor:
                    da = a - n0
                    return a - n0 * (da * n0).sum(-1, keepdim=True) / n0.shape[-1]

                hooks.append((name, ln_jac))
        if level.attention == "frozen":
            for l in range(m.cfg.n_layers):
                hooks.append((f"blocks.{l}.attn.hook_pattern", _freeze(trace.patterns[l])))
        elif level.attention == "jacobian":
            hooks += self._attn_jacobian_hooks(trace)
        store: dict[int, torch.Tensor] = {}  # current transcoder inputs, by source layer
        cur: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # this run's source activations
        for l in range(start, m.cfg.n_layers):
            if level.features != "frozen" and l in self.layers:
                # The transcoder input hook already reflects the level's norm treatment.
                hooks.append((f"blocks.{l}.{self.tc_hook}", _record(store, l)))
            if l not in self.sources:
                continue  # no transcoder writes here: the real MLP stays
            hooks.append(
                (
                    f"blocks.{l}.{self.tc_out_hook}",
                    self._twin_mlp(trace, level.features, l, start, store, cur, by_layer),
                )
            )
        return hooks

    def _attn_jacobian_hooks(self, trace: CleanTrace) -> list:
        """First-order expansion of attention around the clean run:
        ds = (dq . k0 + q0 . dk) / sqrt(d_head),   dp = p0 * (ds - sum_k p0 ds),
        z  = p0 v + dp v0      (the second-order term dp dv is removed).
        With a score softcap s -> c tanh(s / c) (Gemma-2), ds is first multiplied by its
        derivative at the clean score, 1 - tanh^2(s0 / c)."""
        hooks: list = []
        cfg = self.model.cfg
        scale = getattr(cfg, "query_pre_attn_scalar", cfg.d_head) ** 0.5
        cap = getattr(cfg, "attn_softcap", None)
        for l in range(cfg.n_layers):
            q0, k0, v0 = trace.qkv[l]
            p0 = trace.patterns[l]
            if cap is not None:
                s0 = torch.einsum("bqhe,bkhe->bhqk", q0, k0) / scale
                dcap0 = 1 - torch.tanh(s0 / cap) ** 2
            else:
                dcap0 = None
            cur: dict[str, torch.Tensor] = {}
            for x in "qkv":
                hooks.append((f"blocks.{l}.attn.hook_{x}", _record(cur, x)))

            def pattern(
                _p: torch.Tensor, _: str, c=cur, q0=q0, k0=k0, p0=p0, dcap=dcap0
            ) -> torch.Tensor:
                dq, dk = c["q"] - q0, c["k"] - k0
                ds = (
                    torch.einsum("bqhe,bkhe->bhqk", dq, k0)
                    + torch.einsum("bqhe,bkhe->bhqk", q0, dk)
                ) / scale
                if dcap is not None:
                    ds = ds * dcap
                c["dp"] = p0 * (ds - (p0 * ds).sum(-1, keepdim=True))
                return p0 + c["dp"]

            def z(zt: torch.Tensor, _: str, c=cur, v0=v0) -> torch.Tensor:
                return zt - torch.einsum("bhqk,bkhe->bqhe", c["dp"], c["v"] - v0)

            hooks.append((f"blocks.{l}.attn.hook_pattern", pattern))
            hooks.append((f"blocks.{l}.attn.hook_z", z))
        return hooks

    def _twin_mlp(
        self,
        trace: CleanTrace,
        features: str,
        l: int,
        start: int,
        store: dict[int, torch.Tensor],
        cur: dict[int, tuple[torch.Tensor, torch.Tensor]],
        by_layer: dict[int, list[tuple[int, FeatureEdit]]],
    ):
        """Replacement for ``hook_mlp_out`` at output layer ``l``, computed sparsely:
        * frozen: downstream features keep their clean values -> clean output plus edit deltas;
        * gated: the ReLU gates are frozen -> only features active in the clean run matter;
        * relu: full encode (new features may turn on), then a sparse decode.

        Sources below ``start`` (the first edited layer) read an unperturbed input, so their
        clean contribution is used. Each source's activations are computed once per run, at its
        own layer (``cur``), and reused by every later output layer it writes to.
        """
        if features == "frozen":
            edit_deltas = self._edit_deltas(trace, l, by_layer)
            return lambda out, _: edit_deltas(trace.mlp_out[l].expand_as(out).clone())

        srcs = self.sources[l]
        fixed = self._sum_sources(
            l,
            [i for i in srcs if i < start],
            lambda i, j: trace.acts[i][:, trace.active[i]] @ trace.dec[i][:, j],
        )
        live = [i for i in srcs if i >= start]
        b_dec = self.tcs[l].b_dec

        def source(i: int) -> tuple[torch.Tensor, torch.Tensor]:
            """(activations [B, pos, k], decoder rows [k, n_out, d_model]) of source ``i``."""
            if i in cur:
                return cur[i]
            tc, items = self.tcs[i], by_layer.get(i, [])
            if features == "gated":
                active = set(trace.active[i].tolist())
                extra = [e.feature for _, e in items if e.feature not in active]
                idx = torch.cat([trace.active[i], torch.tensor(extra, dtype=torch.long)])
                W_enc, b_enc = tc.enc_rows(idx)  # [k, d_model], [k]
                W_dec = trace.dec[i]
                if extra:
                    W_dec = torch.cat([W_dec, self._dec_rows(i, extra)])
                gate = (trace.acts[i][:, idx] > 0).float()
                col = {int(f): c for c, f in enumerate(idx.tolist())}
                acts = (store[i] @ W_enc.T + b_enc) * gate
                for row, e in items:
                    acts[row, e.pos, col[e.feature]] = e.value
            else:
                acts = tc.encode(store[i])
                if self.skip_first:
                    # trace() put the whole position-0 output in the error term; re-encoding
                    # it here would count it twice.
                    acts[:, 0] = 0
                for row, e in items:
                    acts[row, e.pos, e.feature] = e.value
                idx = acts.flatten(0, 1).any(0).nonzero()[:, 0]
                acts, W_dec = acts[..., idx], self._dec_rows(i, idx)
            cur[i] = (acts, W_dec)
            return cur[i]

        def twin_fn(out: torch.Tensor, _: str) -> torch.Tensor:
            def term(i: int, j: int) -> torch.Tensor:
                acts, W_dec = source(i)
                return acts @ W_dec[:, j]

            total = self._sum_sources(l, live, term)
            if fixed is not None:
                total = fixed if total is None else fixed + total
            return total + b_dec + trace.error[l]

        return twin_fn

    # -------------------------------------------------------------- attribution
    def feature_label(self, layer: int, feature: int, k: int = 6) -> list[str]:
        """Offline label: the tokens whose logits the feature's decoder direction raises most
        (a logit-lens reading of ``W_dec[f]``, summed over its output layers for a CLT)."""
        key = (layer, feature)
        if key not in self._labels:
            direction = self._dec_rows(layer, [feature])[0].sum(0) * self.model.ln_f[0]
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

        def leaf_mlp(l: int) -> HookFn:
            def hook(x: torch.Tensor, _: str) -> torch.Tensor:
                total = self._sum_sources(
                    l, self.sources[l], lambda i, j: leaves[i] @ trace.dec[i][:, j]
                )
                return (total + self.tcs[l].b_dec + trace.error[l][0])[None]

            return hook

        hooks = [(n, _freeze(s)) for n, s in trace.scales.items()]
        for l in range(m.cfg.n_layers):
            hooks.append((f"blocks.{l}.attn.hook_pattern", _freeze(trace.patterns[l])))
        for l in self.out_layers:
            hooks.append((f"blocks.{l}.{self.tc_out_hook}", leaf_mlp(l)))
        with torch.enable_grad(), m.hooks(hooks):
            logits = m(trace.tokens, last_only=True)
            return target_fn(logits), leaves
