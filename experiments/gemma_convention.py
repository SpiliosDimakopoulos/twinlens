"""Which hooks do the Gemma Scope transcoders read and write? (DECISIONS.md, D-Gemma)

    python experiments/gemma_convention.py [--layers 2 8 14 20]

For each of the four input/output conventions and each layer, on one paragraph (BOS excluded):

* FVU: ||y - tc(x)||² / ||y - mean(y)||² with x, y read at the candidate hooks;
* splice recovery: the transcoder output (no error term) replaces the candidate output hook at
  that one layer; recovery = (CE_zero - CE_splice) / (CE_zero - CE_clean), where CE_zero zeroes
  the same hook. 1 = as good as the real MLP, 0 = as bad as deleting it.

The right convention has FVU well below 1 and recovery near 1; a wrong input scale (with/without
the ``1 + weight`` factor) shows up as a large FVU and a distorted L0.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from common import ROOT  # noqa: F401  (puts the repo on sys.path)

from twinlens.core.gemma2 import Gemma2
from twinlens.twin.transcoders import GEMMA_L0, GemmaScopeTranscoderSet

PARAGRAPH = (
    "The river had flooded twice that spring, and by May the farmers along its banks had given "
    "up on planting the low fields. Instead they moved their cattle to the hills, where the "
    "grass came late but thick, and spent the long evenings repairing fences that the water had "
    "carried away. Nobody in the village could remember a year like it, although the oldest "
    "records in the church mentioned a similar flood in 1847."
)
INPUTS = ("hook_mlp_in", "ln2.hook_normalized")  # with / without the (1 + weight) factor
OUTPUTS = ("hook_mlp_out", "mlp.hook_out")  # after / before post_feedforward_layernorm


def ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    return F.cross_entropy(logits[0, :-1], tokens[0, 1:]).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20])
    args = ap.parse_args()
    model = Gemma2.load()
    tcs = GemmaScopeTranscoderSet(args.layers)
    tokens = model.to_tokens(PARAGRAPH)
    names = [f"blocks.{l}.{h}" for l in args.layers for h in (*INPUTS, *OUTPUTS)]
    logits, cache = model.run_with_cache(tokens, names)
    ce_clean = ce(logits, tokens)
    print(f"{tokens.shape[1]} tokens, clean CE {ce_clean:.3f}\n")
    print(
        f"{'layer':>5} {'input':>20} {'output':>13} {'L0':>7} {'L0 ref':>6} {'FVU':>7} {'recov':>7}"
    )
    fvus: dict[tuple[str, str], list[float]] = {}
    for l in args.layers:
        tc = tcs[l]
        for out_hook in OUTPUTS:
            name = f"blocks.{l}.{out_hook}"
            with model.hooks([(name, lambda x, _: torch.zeros_like(x))]):
                ce_zero = ce(model(tokens), tokens)
            for in_hook in INPUTS:
                acts = tc.encode(cache[f"blocks.{l}.{in_hook}"][0])
                recon = tc.decode(acts)
                y = cache[name][0]
                fvu = ((y - recon)[1:].pow(2).sum() / (y[1:] - y[1:].mean(0)).pow(2).sum()).item()
                l0 = (acts[1:] > 0).float().sum(-1).mean().item()
                spliced = recon[None].clone()
                spliced[:, 0] = y[0]  # BOS keeps its real output
                with model.hooks([(name, lambda x, _, s=spliced: s.expand_as(x))]):
                    ce_splice = ce(model(tokens), tokens)
                recov = (ce_zero - ce_splice) / (ce_zero - ce_clean)
                fvus.setdefault((in_hook, out_hook), []).append(fvu)
                print(
                    f"{l:>5} {in_hook:>20} {out_hook:>13} {l0:7.1f} {GEMMA_L0[l]:>6} "
                    f"{fvu:7.3f} {recov:7.3f}",
                    flush=True,
                )
    # The convention with the lowest mean FVU, as JSON for the sweep (--tc-in/--tc-out).
    best = min(fvus, key=lambda k: sum(fvus[k]) / len(fvus[k]))
    print(json.dumps({"tc_in": best[0], "tc_out": best[1],
                      "mean_fvu": {f"{i}|{o}": sum(v) / len(v) for (i, o), v in fvus.items()}}))


if __name__ == "__main__":
    main()
