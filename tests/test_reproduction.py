"""Reproduction suite: published GPT-2 small findings must come out of our own implementation.

If any of these fail, measurements made by the rest of the system cannot be trusted either.
"""

import torch

from twinlens.causal.patching import causal_trace, logit_diff_metric, patch_heads
from twinlens.circuits import known
from twinlens.circuits.datasets import ioi_batch, known_facts


def test_induction_heads_olsson_2022(model):
    scores = known.induction_scores(model)
    top5 = set(known.top_k(scores, 5))
    assert top5 == known.INDUCTION_HEADS
    assert min(scores[l, h] for l, h in known.INDUCTION_HEADS) > 0.7


def test_ioi_behaviour(model):
    batch = ioi_batch(model, n=32)
    assert known.ioi_logit_diff(model, batch) > 2.0
    abc = ioi_batch(model, n=32)
    abc.tokens = abc.corrupt_tokens
    assert abs(known.ioi_logit_diff(model, abc)) < 0.5


def test_ioi_name_movers_by_dla_wang_2022(model):
    dla = known.ioi_head_dla(model, ioi_batch(model, n=32))
    assert set(known.top_k(dla, 3)) == known.NAME_MOVER_HEADS
    assert set(known.top_k(dla, 2, largest=False)) == known.NEGATIVE_NAME_MOVERS


def test_ioi_name_mover_ablation_is_specific(model):
    batch = ioi_batch(model, n=32)
    base = known.ioi_logit_diff(model, batch)
    ablated = known.ioi_ablate_heads(model, batch, known.NAME_MOVER_HEADS)
    control = known.ioi_ablate_heads(model, batch, {(1, 3), (2, 7), (4, 4)})
    # Backup name movers compensate partially (Wang et al. 2022), so the drop is ~30%, not 100%.
    assert ablated < 0.8 * base
    assert abs(control - base) < 0.1 * base


def test_ioi_head_patching_finds_name_movers(model):
    batch = ioi_batch(model, n=8)
    grid = patch_heads(
        model, batch.tokens, batch.corrupt_tokens, logit_diff_metric(batch.io, batch.s)
    )
    assert known.top_k(grid, 1)[0] == (9, 9)
    assert set(known.top_k(grid, 4)) >= {(9, 9), (9, 6)}


def test_factual_recall_causal_tracing_meng_2022(model):
    facts = known_facts(model)[:3]
    assert len(facts) == 3
    late_last, early_subj, attn_last_late = [], [], []
    for f in facts:
        r = causal_trace(model, f.tokens, f.subject_positions, f.target, kind="resid", n_samples=4)
        e = (r["effect"] - r["p_corrupt"]) / (r["p_clean"] - r["p_corrupt"])
        last_subj = f.subject_positions[-1]
        early_subj.append(e[:6, last_subj].mean())
        late_last.append(e[10:, -1].mean())
        a = causal_trace(
            model, f.tokens, f.subject_positions, f.target, kind="attn", window=3, n_samples=4
        )
        ea = (a["effect"] - a["p_corrupt"]) / (a["p_clean"] - a["p_corrupt"])
        attn_last_late.append(ea[7:, -1].max() - ea[:7, -1].max())
    # "Early site": restoring the last subject token in early/mid layers recovers a large share.
    assert torch.stack(early_subj).mean() > 0.15
    # "Late site": restoring the final token late in the network recovers most of it.
    assert torch.stack(late_last).mean() > 0.5
    # Attention's effect at the last token is concentrated in later layers.
    assert torch.stack(attn_last_late).mean() > 0.1
