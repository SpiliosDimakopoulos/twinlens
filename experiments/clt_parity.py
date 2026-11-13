"""Parity check: the twin's feature -> logit direct effects vs. circuit-tracer's graph edges,
with the same cross-layer transcoder (plan task 0.6, DECISIONS.md D5).

    python experiments/clt_parity.py twin      # our effects  -> results/clt_parity/twin.pt
    python experiments/clt_parity.py ct        # circuit-tracer edges -> results/clt_parity/ct.pt
    python experiments/clt_parity.py compare   # Pearson per prompt -> summary.json["clt_parity"]

The stages run as separate processes so that only one copy of Gemma-2-2B is in memory. The ``ct``
stage needs circuit-tracer installed (its nnsight backend loads the ungated
``unsloth/gemma-2-2b`` mirror directly). Both sides use the conventions of circuit-tracer's
graph (D5): the target is the *demeaned* logit of the clean top-1 token, the final softcap is
off, BOS features are zero, attention patterns and norm scales are frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from common import RESULTS, ROOT

N_PROMPTS = 3
OUT = RESULTS / "clt_parity"
CLT_DIR = None  # default: CrossLayerTranscoderSet's cache directory


def _prompts() -> list[tuple[str, str]]:
    corpus = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    return [(cat, ps[0]) for cat, ps in list(corpus.items())[:N_PROMPTS]]


def stage_twin() -> None:
    from twinlens.core.gemma2 import Gemma2
    from twinlens.twin.replacement import Twin
    from twinlens.twin.transcoders import CrossLayerTranscoderSet

    model = Gemma2.load()
    model.final_softcap = None
    twin = Twin(model, CrossLayerTranscoderSet(26, CLT_DIR))
    out = []
    for cat, text in _prompts():
        tokens = model.to_tokens(text)
        trace = twin.trace(tokens)
        target = int(trace.logits[-1].argmax())
        logit, leaves = twin._leaf_forward(
            trace, lambda lg, t=target: lg[0, -1, t] - lg[0, -1].mean()
        )
        grads = torch.autograd.grad(logit, list(leaves.values()))
        keys, vals = [], []
        for (l, leaf), g in zip(leaves.items(), grads, strict=True):
            eff = g * leaf.detach()  # [pos, k]
            pos, col = eff.nonzero(as_tuple=True)
            feats = trace.active[l][col]
            keys += [(l, int(p), int(f)) for p, f in zip(pos.tolist(), feats.tolist(), strict=True)]
            vals += eff[pos, col].tolist()
        out.append(
            {
                "category": cat,
                "tokens": tokens[0].tolist(),
                "target": target,
                "keys": keys,
                "effects": vals,
            }
        )
        print(f"twin {cat}: {len(keys)} features, target {target}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / "twin.pt")


def stage_ct() -> None:
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

    from twinlens.twin.transcoders import CrossLayerTranscoderSet

    ours = torch.load(OUT / "twin.pt")
    clt_dir = str(CrossLayerTranscoderSet(26, CLT_DIR).dir)
    cpu = torch.device("cpu")
    clt = load_clt(clt_dir, device=cpu, dtype=torch.float32, lazy_encoder=True, lazy_decoder=True)
    model = ReplacementModel.from_pretrained_and_transcoders(
        "unsloth/gemma-2-2b", clt, backend="nnsight", device=cpu, dtype=torch.float32
    )
    out = []
    for p in ours:
        tokens = torch.tensor(p["tokens"])
        graph = attribute(
            tokens,
            model,
            attribution_targets=torch.tensor([p["target"]]),
            batch_size=64,
            max_feature_nodes=8,  # only the logit row is needed: it covers every feature
        )
        n = len(graph.active_features)
        row = graph.adjacency_matrix[-1, :n].float()  # the single logit node
        keys = [tuple(int(x) for x in af) for af in graph.active_features.tolist()]
        out.append({"keys": keys, "edges": row.tolist()})
        print(f"ct {p['category']}: {n} active features", flush=True)
    torch.save(out, OUT / "ct.pt")


def stage_compare() -> None:
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import update_summary

    ours, theirs = torch.load(OUT / "twin.pt"), torch.load(OUT / "ct.pt")
    per_prompt = []
    for a, b in zip(ours, theirs, strict=True):
        ea = dict(zip(map(tuple, a["keys"]), a["effects"], strict=True))
        eb = dict(zip(map(tuple, b["keys"]), b["edges"], strict=True))
        keys = sorted(set(ea) | set(eb))
        x = torch.tensor([ea.get(k, 0.0) for k in keys], dtype=torch.float64)
        y = torch.tensor([eb.get(k, 0.0) for k in keys], dtype=torch.float64)
        r = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        top = y.abs().argsort(descending=True)[:20]
        per_prompt.append(
            {
                "category": a["category"],
                "n_ours": len(ea),
                "n_theirs": len(eb),
                "n_only_ours": len(set(ea) - set(eb)),
                "n_only_theirs": len(set(eb) - set(ea)),
                "pearson": r,
                "pearson_top20": torch.corrcoef(torch.stack([x[top], y[top]]))[0, 1].item(),
                "max_abs_diff": (x - y).abs().max().item(),
                "max_abs_edge": y.abs().max().item(),
            }
        )
        print(json.dumps(per_prompt[-1]), flush=True)
    update_summary(
        {
            "clt_parity": {
                "prompts": per_prompt,
                "min_pearson": min(p["pearson"] for p in per_prompt),
            }
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["twin", "ct", "compare"])
    ap.add_argument("--clt-dir", type=Path, default=None)
    args = ap.parse_args()
    CLT_DIR = args.clt_dir
    {"twin": stage_twin, "ct": stage_ct, "compare": stage_compare}[args.stage]()
