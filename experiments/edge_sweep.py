"""Feature-to-feature edge sweep (plan phase 1, DECISIONS.md D8).

    python experiments/edge_sweep.py --model gpt2                       # GPT-2, PLT
    python experiments/edge_sweep.py --model gemma2 --n-top 3 --n-random 3
    python experiments/edge_sweep.py --model gemma2 --transcoders clt426k --n-top 3 --n-random 3
    python experiments/edge_sweep.py --small ...                        # 1 prompt per family

The edits are those of ``fidelity_sweep.py`` with the same flags (same sampler and seed per
prompt). Per prompt, rows go to ``results/raw_edges[_small]_<tag>/<id>.parquet`` (resumable);
the concatenation is ``results/edges[_small]_<tag>.parquet``, one row per (edit, target).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from common import RESULTS, ROOT, write_parquet

from twinlens.fidelity.edges import run_prompt_edges
from twinlens.twin.replacement import Twin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=12, help="edits per batched forward")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--model", choices=["gpt2", "gemma2"], default="gpt2")
    ap.add_argument("--transcoders", choices=["plt", "clt426k"], default="plt")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    corpus = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    prompts = [(cat, p) for cat, ps in corpus.items() for p in ps[: 1 if args.small else None]]
    if args.model == "gpt2":
        from twinlens.core.gpt2 import GPT2

        twin, tag = Twin(GPT2.load("gpt2")), "gpt2"
    else:
        from twinlens.core.gemma2 import Gemma2
        from twinlens.twin.transcoders import CrossLayerTranscoderSet, GemmaScopeTranscoderSet

        if args.transcoders == "clt426k":
            twin, tag = Twin(Gemma2.load(), CrossLayerTranscoderSet(26)), "gemma2_clt"
        else:  # the hooks of the Gemma PLT output sweep (DECISIONS.md D4)
            tcs = GemmaScopeTranscoderSet(range(26), "ln2.hook_normalized", "hook_mlp_out")
            twin, tag = Twin(Gemma2.load(), tcs), "gemma2"
    small = "_small" if args.small else ""
    out = Path(args.out) / f"raw_edges{small}_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i, (cat, text) in enumerate(prompts):
        path = out / f"{i:04d}.parquet"
        if path.exists():
            continue
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        t = time.perf_counter()
        rows = run_prompt_edges(
            twin, text, cat, i, args.n_top, args.n_random, chunk=args.chunk, seed=i
        )
        write_parquet(pd.DataFrame(rows), path)
        print(
            f"[{i + 1}/{len(prompts)}] {cat:10s} {len(rows)} pairs "
            f"{time.perf_counter() - t:.0f}s  total {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    df = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))], ignore_index=True)
    name = f"edges{small}_{tag}.parquet"
    write_parquet(df, Path(args.out) / name)
    print(f"wrote {len(df)} rows -> {Path(args.out) / name}")


if __name__ == "__main__":
    main()
