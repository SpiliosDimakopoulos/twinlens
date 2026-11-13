"""Attribution-patching baseline: one backward pass through the *real* GPT-2 per prompt.

    python experiments/atp_baseline.py

For every edit in ``results/fidelity.parquet`` it predicts the effect of moving the feature's
decoder direction, ``Δlog p ≈ ∂log p(target)/∂mlp_out[l, pos] · (a' − a) W_dec[f]``, with the
gradient taken through the real model (LayerNorm and attention are *not* frozen). This is the
first-order estimate attribution patching (Nanda 2023; Syed et al. 2024; Kramár et al. 2024)
gives for the same intervention. It answers the review question: how much of Fix B's accuracy
does a plain backward pass through the real model already give?

Writes ``results/fidelity_atp.parquet`` (one ``atp_dlogp`` per edit, aligned by row, with the
edit key columns so analyze_atp.py can check the alignment).
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from common import EDIT_KEY, RESULTS, write_parquet

from twinlens.core.gpt2 import GPT2


def mlp_out_grads(
    model: GPT2, text: str, hook: str = "hook_mlp_out"
) -> tuple[int, dict[int, torch.Tensor]]:
    """Clean top-1 target and ∂log p(target)/∂blocks.{l}.{hook} for every layer, [pos, d_model].
    ``hook`` is where the transcoder's output (and so the feature edit) lives."""
    toks = model.to_tokens(text)
    store: dict[int, torch.Tensor] = {}

    def make(layer: int):
        def fn(x: torch.Tensor, _: str) -> torch.Tensor:
            x = x.detach().requires_grad_(True)
            store[layer] = x
            return x

        return fn

    hooks = [(f"blocks.{l}.{hook}", make(l)) for l in range(model.cfg.n_layers)]
    with torch.enable_grad(), model.hooks(hooks):
        logits = model(toks)[0, -1]
        target = int(logits.argmax())
        logits.log_softmax(-1)[target].backward()
    return target, {l: x.grad[0].detach() for l, x in store.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2", "gemma2"], default="gpt2")
    ap.add_argument("--tc-in", default="ln2.hook_normalized", help="gemma2: transcoder input hook")
    ap.add_argument("--tc-out", default="hook_mlp_out", help="gemma2: transcoder output hook")
    args = ap.parse_args()
    tag = "" if args.model == "gpt2" else f"_{args.model}"
    df = pd.read_parquet(RESULTS / f"fidelity{tag}.parquet")
    if args.model == "gpt2":
        from twinlens.twin.transcoders import TranscoderSet

        model, tcs, hook = GPT2.load("gpt2"), TranscoderSet(12), "hook_mlp_out"
    else:
        from twinlens.core.gemma2 import Gemma2
        from twinlens.twin.transcoders import GemmaScopeTranscoderSet

        model, hook = Gemma2.load(), args.tc_out
        tcs = GemmaScopeTranscoderSet(range(26), in_hook=args.tc_in, out_hook=args.tc_out)
    out = pd.Series(index=df.index, dtype=float)
    for pid, g in df.groupby("prompt_id"):
        target, grads = mlp_out_grads(model, g.prompt.iloc[0], hook)
        for i, r in g.iterrows():
            l, p, f = int(r.layer), int(r.pos), int(r.feature)
            direction = tcs[l].dec_rows([f])[0].float()
            out[i] = float(r.delta_act * (grads[l][p] @ direction))
        print(f"prompt {pid}: {len(g)} edits, target id {target}", flush=True)
    res = df[list(EDIT_KEY)].assign(atp_dlogp=out)
    path = RESULTS / f"fidelity_atp{tag}.parquet"
    write_parquet(res, path)
    print(f"wrote {len(res)} rows -> {path}")

    # Sanity check: on the gradient path, AtP and the twin's direct edge share the final
    # LayerNorm / unembedding; their sign should agree on most edits.
    agree = ((res.atp_dlogp > 0) == (df.direct_dlogp > 0)).mean()
    print(f"sign agreement AtP vs direct edge: {agree:.3f}")


if __name__ == "__main__":
    main()
