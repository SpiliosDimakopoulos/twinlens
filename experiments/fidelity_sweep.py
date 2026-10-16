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
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinlens.core.gpt2 import GPT2  # noqa: E402
from twinlens.fidelity.engine import run_prompt  # noqa: E402
from twinlens.twin.replacement import Twin  # noqa: E402


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
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results"))
    args = ap.parse_args()

    corpus = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    prompts = [(cat, p) for cat, ps in corpus.items() for p in (ps[:1] if args.small else ps)]
    out = Path(args.out) / ("raw_small" if args.small else "raw")
    out.mkdir(parents=True, exist_ok=True)

    twin = Twin(GPT2.load("gpt2"))
    t0 = time.perf_counter()
    for i, (cat, text) in enumerate(prompts):
        path = out / f"{i:04d}.parquet"
        if path.exists():
            continue
        t = time.perf_counter()
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        rows = run_prompt(twin, text, cat, i, args.n_top, args.n_random, chunk=args.chunk, seed=i)
        pd.DataFrame(rows).to_parquet(path)
        print(
            f"[{i + 1}/{len(prompts)}] {cat:10s} {len(rows)} edits "
            f"{time.perf_counter() - t:.0f}s  total {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    df = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))], ignore_index=True)
    name = "fidelity_small.parquet" if args.small else "fidelity.parquet"
    df.to_parquet(Path(args.out) / name)
    print(f"wrote {len(df)} rows -> {Path(args.out) / name}")


if __name__ == "__main__":
    main()
