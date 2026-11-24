"""Batched feature interventions with the semantics of circuit-tracer's ``feature_intervention``.

``feature_intervention`` (no ``constrained_layers``) runs one intervention set per forward pass,
computes the logits of every position, and with ``freeze_attention=True`` reruns the clean
prompt on every call to cache the attention patterns. On Gemma-2-2B that made D10 take about
110 minutes per prompt (DECISIONS.md D10, deviation of 2026-11-24). This class does the same
computation with three changes:
- many intervention sets in one batched forward pass;
- the unembedding of the last position only;
- clean attention patterns cached once per prompt.

The semantics follow circuit-tracer, so the results are the same; ``tests/test_ct.py`` checks
this against ``feature_intervention``. At every layer, the live activations of the candidate
features are the transcoder's ``encode_layer`` of the live input, with the special positions
zeroed. A candidate set to value v changes by Δ = v − live activation. Each output layer it
writes to receives Δ · W_dec, added to the real MLP output at the feature-output hook.
Everything else is recomputed, with attention live (T1) or frozen at the clean patterns (T2).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

Feature = tuple[int, int, int]


class Intervener:
    def __init__(self, model, prompt, feats: Sequence[Feature], freeze_attention: bool = False):
        self.model = model
        self.tokens = model.ensure_tokenized(prompt).reshape(1, -1)
        self.n_pos = self.tokens.shape[1]
        self.feats = list(feats)
        self.freeze_attention = freeze_attention
        self.n_layers = model.cfg.n_layers
        self.layers: dict[int, dict] = {}
        for i, (l, p, f) in enumerate(self.feats):
            self.layers.setdefault(l, {"idx": [], "pos": [], "feat": []})
            g = self.layers[l]
            g["idx"].append(i)
            g["pos"].append(p)
            g["feat"].append(f)
        for l, g in self.layers.items():
            g["idx"] = torch.tensor(g["idx"])
            g["pos"] = torch.tensor(g["pos"])
            g["feat"] = torch.tensor(g["feat"])
            dec = model.transcoders._get_decoder_vectors(l, g["feat"]).float()
            g["dec"] = dec if dec.ndim == 3 else dec[:, None, :]  # [n, n_out_layers, d]
        self.zero_positions = model.zero_positions
        self.patterns = None
        if freeze_attention:
            names = [f"blocks.{l}.attn.hook_pattern" for l in range(self.n_layers)]
            with torch.no_grad():
                _, cache = model.run_with_cache(self.tokens, names_filter=lambda n: n in names)
            self.patterns = {n: cache[n] for n in names}

    @torch.no_grad()
    def run(self, values: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        """Last-position logits for each row of ``values`` [B, n_feats]. Row b sets feature i to
        ``values[b, i]``; NaN leaves it alone. Returns [B, d_vocab]."""
        out = [self._run(values[s : s + chunk]) for s in range(0, len(values), chunk)]
        return torch.cat(out)

    def _run(self, values: torch.Tensor) -> torch.Tensor:
        m = self.model
        B = values.shape[0]
        values = values.float()
        deltas = torch.zeros(B, self.n_layers, self.n_pos, m.cfg.d_model)

        def at_input(x, hook, layer):
            g = self.layers[layer]
            v = values[:, g["idx"]]  # [B, n]
            if torch.isnan(v).all():
                return x
            acts = m.transcoders.encode_layer(x.float(), layer)  # [B, pos, F]
            acts[:, self.zero_positions] = 0
            live = acts[:, g["pos"], g["feat"]]  # [B, n]
            d = torch.where(torch.isnan(v), torch.zeros_like(v), v - live)
            n, n_out, dm = g["dec"].shape
            # feature i adds d[b, i] * dec[i, j] to output layer layer + j at its position:
            # sum over features first, as one matmul, so no [B, n_out, n, d] tensor is built
            onehot = torch.nn.functional.one_hot(g["pos"], self.n_pos).float()  # [n, pos]
            w = (d[:, :, None] * onehot[None]).transpose(1, 2)  # [B, pos, n]
            add = (w @ g["dec"].reshape(n, n_out * dm)).reshape(B, self.n_pos, n_out, dm)
            deltas[:, layer : layer + n_out] += add.transpose(1, 2)
            return x

        def at_output(x, hook, layer):
            return x + deltas[:, layer].to(x.dtype)

        def freeze(x, hook):
            return self.patterns[hook.name].expand_as(x)

        hooks = [(f"blocks.{l}.{m.feature_input_hook}", _bind(at_input, l)) for l in self.layers]
        hooks += [
            (f"blocks.{l}.{m.feature_output_hook}", _bind(at_output, l))
            for l in range(self.n_layers)
        ]
        if self.patterns is not None:
            hooks += [(n, freeze) for n in self.patterns]
        tokens = self.tokens.expand(B, -1)
        with m.hooks(fwd_hooks=hooks):
            resid = m(tokens, stop_at_layer=self.n_layers)
        logits = m.unembed(m.ln_final(resid[:, -1:]))[:, -1]
        cap = getattr(m.cfg, "output_logits_soft_cap", 0.0) or 0.0
        if cap > 0:
            logits = cap * torch.tanh(logits / cap)
        return logits.float()


def _bind(fn, layer):
    def hook(x, hook):
        return fn(x, hook, layer)

    return hook
