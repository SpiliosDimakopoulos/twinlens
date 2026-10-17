"""Prepare the Qwen3-0.6B transcoders for the study prompts with minimal disk use.

    python experiments/qwen_prepare.py [--max-minutes 8]

For each twin layer: download the encoder (~335 MB, kept), find every feature active on any
study prompt, download the decoder temporarily, cache just those rows, delete the decoder.
Peak disk ≈ encoders (18 × 335 MB) + one decoder. Resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinlens.core.qwen3 import Qwen3  # noqa: E402
from twinlens.twin.transcoders import CACHE_DIR, RemoteRowTranscoder  # noqa: E402

LAYERS = range(10, 28)
STATE = CACHE_DIR / "twinlens" / "qwen-tc" / "prepared.json"


def study_prompts() -> list[tuple[str, str]]:
    corpus = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    return [(cat, p) for cat, ps in corpus.items() for p in ps[:2]]  # 2 per family


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    done = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    model = Qwen3.load()
    names = [f"blocks.{l}.hook_mlp_in" for l in LAYERS]
    inputs = [
        model.run_with_cache(model.to_tokens(p), names, last_only=True)[1]
        for _, p in study_prompts()
    ]
    for layer in LAYERS:
        if layer in done:
            continue
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        t = time.perf_counter()
        tc = RemoteRowTranscoder(layer)
        active = set()
        for cache in inputs:
            acts = torch.relu(tc.pre(cache[f"blocks.{layer}.hook_mlp_in"][0, 1:]))  # skip the sink
            active |= set(acts.any(0).nonzero()[:, 0].tolist())
        tc.prefetch_decoder()
        tc.dec_rows(sorted(active))
        tc.drop_decoder()
        done.add(layer)
        STATE.write_text(json.dumps(sorted(done)))
        print(
            f"L{layer}: {len(active)} active features cached, {time.perf_counter() - t:.0f}s",
            flush=True,
        )
    print("all layers prepared", flush=True)


if __name__ == "__main__":
    main()
