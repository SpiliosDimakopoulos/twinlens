"""Download Gemma-2-2B and its Gemma Scope transcoders and convert them to memory-mapped F32.

    python experiments/gemma_prepare.py [--layers 0-25] [--keep-npz] [--max-minutes 0]

Model: ``unsloth/gemma-2-2b`` (5.2 GB BF16) -> ~10.5 GB F32. Transcoders: one ``params.npz``
(302 MB) per layer -> F32 safetensors; the npz is deleted after conversion unless
``--keep-npz``. Peak disk ≈ 26 GB, peak RAM well under 1 GB. Resumable: finished files are
skipped and partial conversions are redone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import ROOT  # noqa: F401  (puts the repo on sys.path)

from twinlens.core.gemma2 import Gemma2
from twinlens.core.weights import hf_file, npz_to_safetensors
from twinlens.twin.transcoders import GEMMA_KEYS, GEMMA_L0, GEMMA_REPO, GemmaScopeTranscoderSet


def _layers(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-25")
    ap.add_argument("--keep-npz", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    model = Gemma2.load()  # downloads + converts the model on first run
    print(f"model ready: {model.cfg.n_layers} layers, {time.perf_counter() - t0:.0f}s", flush=True)
    tcs = GemmaScopeTranscoderSet()
    for layer in _layers(args.layers):
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        t = time.perf_counter()
        dst = tcs.dir / f"layer_{layer}.f32.safetensors"
        if dst.exists():
            continue
        name = f"layer_{layer}/width_16k/average_l0_{GEMMA_L0[layer]}/params.npz"
        npz = hf_file(GEMMA_REPO, name)
        npz_to_safetensors(npz, dst, GEMMA_KEYS)
        if not args.keep_npz:
            blob = Path(npz).resolve()  # the Hub cache keeps the bytes in a blob behind a link
            Path(npz).unlink(missing_ok=True)
            blob.unlink(missing_ok=True)
        tc = tcs[layer]
        print(
            f"L{layer}: d_sae {tc.d_sae}, threshold min {float(tc.threshold.min()):.3g}, "
            f"{time.perf_counter() - t:.0f}s",
            flush=True,
        )
    print("all layers prepared", flush=True)


if __name__ == "__main__":
    main()
