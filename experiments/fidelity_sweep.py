"""Run the twin-fidelity sweep over the prompt corpus.

    python experiments/fidelity_sweep.py                 # full sweep (resumable)
    python experiments/fidelity_sweep.py --small         # 1 prompt per category, for CI/demo

Each prompt's rows are written to ``experiments/results/raw/<id>.parquet`` as soon as they are
done, so an interrupted run resumes where it stopped. ``results/fidelity.parquet`` is the
concatenation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from common import RESULTS, ROOT, write_parquet

from twinlens.core.gpt2 import GPT2
from twinlens.fidelity.engine import run_prompt
from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS, Twin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument(
        "--chunk", type=int, default=24, help="edits per batched forward (lower = less RAM)"
    )
    ap.add_argument(
        "--max-minutes", type=float, default=0, help="stop cleanly after this long (resume later)"
    )
    ap.add_argument("--model", choices=["gpt2", "qwen3", "gemma2"], default="gpt2")
    ap.add_argument("--per-family", type=int, default=0, help="prompts per family (0 = all)")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--tc-in", default="hook_mlp_in", help="gemma2: transcoder input hook")
    ap.add_argument("--tc-out", default="hook_mlp_out", help="gemma2: transcoder output hook")
    args = ap.parse_args()

    corpus = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    keep = 1 if args.small else (args.per_family or None)
    prompts = [(cat, p) for cat, ps in corpus.items() for p in ps[:keep]]
    tag = "" if args.model == "gpt2" else f"_{args.model}"
    out = Path(args.out) / (("raw_small" if args.small else "raw") + tag)
    out.mkdir(parents=True, exist_ok=True)

    if args.model == "gpt2":
        twin, modes = Twin(GPT2.load("gpt2")), TWIN_LEVELS
    elif args.model == "gemma2":
        from twinlens.core.gemma2 import Gemma2
        from twinlens.twin.transcoders import GemmaScopeTranscoderSet

        # 16k-feature transcoders: every level is affordable, including the nonlinear ones
        tcs = GemmaScopeTranscoderSet(range(26), in_hook=args.tc_in, out_hook=args.tc_out)
        twin = Twin(Gemma2.load(), tcs)
        modes = (*TWIN_LEVELS, *FIX_LEVELS)
    else:  # nonlinear twins need full 164k-feature encodes per run: too slow on CPU
        from twinlens.core.qwen3 import Qwen3
        from twinlens.twin.transcoders import QwenTranscoderSet

        twin = Twin(Qwen3.load(), QwenTranscoderSet(range(10, 28)))
        modes = ("direct", "linear", "linear_lnjac", "linear_fulljac")
    t0 = time.perf_counter()
    for i, (cat, text) in enumerate(prompts):
        path = out / f"{i:04d}.parquet"
        if path.exists():
            continue
        t = time.perf_counter()
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        rows = run_prompt(
            twin, text, cat, i, args.n_top, args.n_random, chunk=args.chunk, seed=i, modes=modes
        )
        write_parquet(pd.DataFrame(rows), path)  # atomic: a crash never leaves a partial file
        print(
            f"[{i + 1}/{len(prompts)}] {cat:10s} {len(rows)} edits "
            f"{time.perf_counter() - t:.0f}s  total {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    df = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))], ignore_index=True)
    name = ("fidelity_small" if args.small else "fidelity") + tag + ".parquet"
    write_parquet(df, Path(args.out) / name)
    print(f"wrote {len(df)} rows -> {Path(args.out) / name}")


if __name__ == "__main__":
    main()
