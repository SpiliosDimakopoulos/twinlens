"""CPU latency / memory benchmark for the numbers quoted in the README.

    python bench/latency.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinlens.core.gpt2 import GPT2  # noqa: E402
from twinlens.twin.replacement import TWIN_LEVELS, FeatureEdit, Twin  # noqa: E402


def timeit(fn, n: int = 5) -> float:
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def main() -> None:
    proc = psutil.Process(os.getpid())
    t0 = time.perf_counter()
    twin = Twin(GPT2.load("gpt2"))
    m = twin.model
    load_s = time.perf_counter() - t0
    prompt = "When Mary and John went to the store, John gave a drink to"
    toks = m.to_tokens(prompt)
    trace = twin.trace(toks)
    target = int(trace.logits[-1].argmax())
    f = int(trace.acts[6][-2].argmax())
    edits = [FeatureEdit(6, toks.shape[1] - 2, f, 0.0)]
    res = {
        "threads": torch.get_num_threads(),
        "prompt_tokens": toks.shape[1],
        "load_s": round(load_s, 2),
        "forward_ms": round(timeit(lambda: m(toks, last_only=True)), 1),
        "run_with_cache_ms": round(timeit(lambda: m.run_with_cache(toks)), 1),
        "twin_trace_ms": round(timeit(lambda: twin.trace(toks), 3), 1),
        "attribution_ms": round(timeit(lambda: twin.feature_effects(trace, target), 3), 1),
        **{f"whatif_{lv}_ms": round(timeit(lambda lv=lv: twin.run(trace, lv, edits), 3), 1)
           for lv in TWIN_LEVELS},
        "private_memory_gb": round(proc.memory_info().private / 1e9, 2),
    }
    print(json.dumps(res, indent=2))
    (ROOT / "bench" / "latency.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
