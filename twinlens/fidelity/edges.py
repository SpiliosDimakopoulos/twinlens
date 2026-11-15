"""Feature-to-feature edges: does a graph's "feature A moves feature B by w" hold in the real
model? (plan phase 1, DECISIONS.md D8)

For each edit of the output sweep, the downstream targets are clean-active features at a later
layer and the same or a later position. A twin level predicts the change of the target's
pre-activation inside its own run (gates frozen, so also of its activation); the real model
answers with the target's encoder applied to the real transcoder input after the edit, passed
through the transcoder's own nonlinearity.
"""

from __future__ import annotations

import random

import torch

from twinlens.fidelity.engine import EditSpec, sample_edits
from twinlens.twin.replacement import CleanTrace, FeatureEdit, Twin

LEVELS = ("linear", "linear_lnjac", "linear_fulljac")
TRUTHS = ("real", "real_frozen_attn")
K_TOP = 10
K_RANDOM = 10


def _activation(tc, pre: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """The transcoder's own nonlinearity for features ``feats`` (ReLU or JumpReLU)."""
    thr = getattr(tc, "threshold", None)
    if thr is None:
        return torch.relu(pre)
    return pre * (pre > thr[feats].clamp(min=0))


class _Targets:
    """Clean-active candidate targets of one layer (BOS excluded) and their encoder rows."""

    def __init__(self, twin: Twin, trace: CleanTrace, l: int, x0: torch.Tensor):
        acts = trace.acts[l]
        pos, col = acts[1:].nonzero(as_tuple=True)
        self.pos = pos + 1
        self.feat = col
        idx = torch.unique(col)
        W, b = twin.tcs[l].enc_rows(idx)
        where = {int(f): k for k, f in enumerate(idx.tolist())}
        k = torch.tensor([where[int(f)] for f in col.tolist()], dtype=torch.long)
        self.W, self.b = W[k], b[k]  # [n, d_model], [n]: one row per (pos, feature) candidate
        self.act0 = acts[self.pos, self.feat]
        self.pre0 = self.pre(x0)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activations of all candidates from one run's input ``x`` ``[pos, d_model]``."""
        return (x[self.pos] * self.W).sum(-1) + self.b


def run_prompt_edges(
    twin: Twin,
    text: str,
    category: str = "",
    prompt_id: int = 0,
    n_top: int = 2,
    n_random: int = 2,
    chunk: int = 12,
    seed: int = 0,
    **kw,
) -> list[dict]:
    """The edits of the output sweep for this prompt (same sampler and seed), scored on their
    downstream features. Edits in the last layer have no targets and are left out."""
    trace = twin.trace(twin.model.to_tokens(text))
    effects = twin.feature_effects(trace, int(trace.logits[-1].argmax()))
    last = max(twin.layers)
    specs = [
        s for s in sample_edits(twin, trace, effects, n_top, n_random, seed) if s.edit.layer < last
    ]
    rows = score_edits(twin, trace, specs, chunk=chunk, seed=seed, **kw)
    return [{"prompt_id": prompt_id, "category": category, **r} for r in rows]


def score_edits(
    twin: Twin,
    trace: CleanTrace,
    specs: list[EditSpec],
    chunk: int = 12,
    seed: int = 0,
    levels: tuple[str, ...] = LEVELS,
    truths: tuple[str, ...] = TRUTHS,
    k_top: int = K_TOP,
    k_random: int = K_RANDOM,
) -> list[dict]:
    """One row per (edit, target): ``{level}_dact`` predicted by each twin level and
    ``{truth}_dpre`` / ``{truth}_dact`` in the real model. ``levels[0]`` ranks the targets."""
    rng = random.Random(seed)
    clean: dict[int, torch.Tensor] = {}
    twin.run(trace, "real", [None], capture=clean)
    cands = {l: _Targets(twin, trace, l, clean[l][0]) for l in twin.layers}

    rows: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(specs), chunk):
            batch = specs[start : start + chunk]
            edits: list[FeatureEdit | None] = [s.edit for s in batch]
            runs: dict[str, dict[int, torch.Tensor]] = {}
            for mode in (*levels, *truths):
                runs[mode] = {}
                twin.run(trace, mode, edits, capture=runs[mode])
            for i, spec in enumerate(batch):
                e = spec.edit
                # every candidate downstream of the edit, with the graph's prediction
                parts = []
                for l in twin.layers:
                    if l <= e.layer:
                        continue
                    c = cands[l]
                    keep = (c.pos >= e.pos).nonzero()[:, 0]
                    if len(keep):
                        parts.append((l, keep))
                if not parts:
                    continue
                d_graph = torch.cat(
                    [
                        cands[l].pre(runs[levels[0]][l][i])[keep] - cands[l].pre0[keep]
                        for l, keep in parts
                    ]
                )
                owner = torch.cat([torch.full((len(k),), l) for l, k in parts])
                local = torch.cat([k for _, k in parts])
                order = d_graph.abs().argsort(descending=True).tolist()
                top = order[:k_top]
                rest = order[k_top:]
                chosen = [(j, "top") for j in top] + [
                    (j, "random") for j in rng.sample(rest, min(k_random, len(rest)))
                ]
                for j, sel in chosen:
                    l, k = int(owner[j]), int(local[j])
                    c = cands[l]
                    W, b = c.W[k], c.b[k]
                    p, f = int(c.pos[k]), int(c.feat[k])
                    pre0, act0 = float(c.pre0[k]), float(c.act0[k])
                    row = {
                        "layer": e.layer,
                        "pos": e.pos,
                        "feature": e.feature,
                        "kind": spec.kind,
                        "selection": spec.selection,
                        "target_layer": l,
                        "target_pos": p,
                        "target_feature": f,
                        "target_selection": sel,
                        "pre0": pre0,
                        "act0": act0,
                    }
                    for mode in levels:
                        pre = float(runs[mode][l][i, p] @ W + b)
                        row[f"{mode}_dact"] = pre - pre0  # gates frozen, target clean-active
                    for mode in truths:
                        pre_t = (runs[mode][l][i, p] @ W + b)[None]
                        act = _activation(twin.tcs[l], pre_t, torch.tensor([f]))
                        row[f"{mode}_dpre"] = float(pre_t) - pre0
                        row[f"{mode}_dact"] = float(act) - act0
                    rows.append(row)
    return rows
