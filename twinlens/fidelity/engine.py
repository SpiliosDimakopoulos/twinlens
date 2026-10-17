"""Fidelity engine: the twin predicts, the real model answers, and we score the gap.

For one prompt we sample feature interventions, run them through every twin level and through
the real model, and record (a) agreement metrics between each twin level and reality and
(b) cheap *reliability signals* computed from the clean run only, so that we can later test
whether the twin can tell in advance when its own prediction will be wrong.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from twinlens.fidelity.signals import reliability_signals
from twinlens.twin.replacement import TWIN_LEVELS as TWIN_MODES
from twinlens.twin.replacement import CleanTrace, FeatureEdit, Twin


@dataclass
class EditSpec:
    edit: FeatureEdit
    kind: str  # "ablate" | "amplify"
    selection: str  # "top" (largest direct effect) | "random" (random active feature)
    direct_effect: float


def sample_edits(
    twin: Twin,
    trace: CleanTrace,
    effects: dict[int, torch.Tensor],
    n_top: int = 2,
    n_random: int = 2,
    seed: int = 0,
) -> list[EditSpec]:
    """Per layer: the ``n_top`` features with the largest |direct effect| on the predicted logit
    plus ``n_random`` uniformly random active features (to avoid only testing the easy cases).
    Each is ablated and amplified (x2). The BOS position is skipped (its features are outliers).
    """
    rng = random.Random(seed)
    specs: list[EditSpec] = []
    for layer in twin.layers:
        acts = trace.acts[layer][1:]
        active = acts.nonzero()
        if len(active) == 0:
            continue
        eff = effects[layer][1:]
        chosen: list[tuple[int, int, str]] = []
        order = eff.abs().flatten().argsort(descending=True).tolist()
        for i in order:
            p, f = divmod(i, eff.shape[1])
            if acts[p, f] > 0:
                chosen.append((p + 1, f, "top"))
            if len(chosen) == n_top:
                break
        pool = [(int(p) + 1, int(f)) for p, f in active.tolist()]
        taken = {(p, f) for p, f, _ in chosen}
        pool = [x for x in pool if x not in taken]
        for p, f in rng.sample(pool, min(n_random, len(pool))):
            chosen.append((p, f, "random"))
        for p, f, sel in chosen:
            a = trace.acts[layer][p, f].item()
            d = effects[layer][p, f].item()
            specs.append(EditSpec(FeatureEdit(layer, p, f, 0.0), "ablate", sel, d))
            specs.append(EditSpec(FeatureEdit(layer, p, f, 2 * a), "amplify", sel, d))
    return specs


def _effect_metrics(clean: torch.Tensor, out: torch.Tensor, target: int) -> dict[str, torch.Tensor]:
    lp_clean, lp = clean.log_softmax(-1), out.log_softmax(-1)
    return {
        "dlogp": lp[:, target] - lp_clean[target],
        "kl": (lp_clean.exp() * (lp_clean - lp)).sum(-1),
        "top1": out.argmax(-1),
        "dlogits": (out - out.mean(-1, keepdim=True)) - (clean - clean.mean()),
    }


def run_prompt(
    twin: Twin,
    text: str,
    category: str = "",
    prompt_id: int = 0,
    n_top: int = 2,
    n_random: int = 2,
    chunk: int = 24,
    seed: int = 0,
    modes: tuple[str, ...] = TWIN_MODES,
) -> list[dict]:
    model = twin.model
    trace = twin.trace(model.to_tokens(text))
    clean = trace.logits[-1]
    target = int(clean.argmax())
    effects = twin.feature_effects(trace, target)
    specs = sample_edits(twin, trace, effects, n_top, n_random, seed)

    rows: list[dict] = []
    for start in range(0, len(specs), chunk):
        batch = specs[start : start + chunk]
        edits = [s.edit for s in batch]
        outs = {mode: twin.run(trace, mode, edits) for mode in (*modes, "real")}
        met = {mode: _effect_metrics(clean, o, target) for mode, o in outs.items()}
        real = met["real"]
        for i, spec in enumerate(batch):
            row = {
                "prompt_id": prompt_id,
                "category": category,
                "prompt": text,
                "feature": spec.edit.feature,
                "pos": spec.edit.pos,
                "kind": spec.kind,
                "selection": spec.selection,
                **reliability_signals(twin, trace, spec.edit, spec.direct_effect),
                "real_dlogp": real["dlogp"][i].item(),
                "real_kl": real["kl"][i].item(),
                "real_flip": bool(real["top1"][i] != target),
            }
            for mode in modes:
                m = met[mode]
                cos = torch.nn.functional.cosine_similarity(
                    m["dlogits"][i], real["dlogits"][i], dim=0
                ).item()
                row |= {
                    f"{mode}_dlogp": m["dlogp"][i].item(),
                    f"{mode}_kl": m["kl"][i].item(),
                    f"{mode}_flip": bool(m["top1"][i] != target),
                    f"{mode}_cos": cos,
                }
            rows.append(row)
    return rows
