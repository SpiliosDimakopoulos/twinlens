"""Download Gemma-2-2B and the 426K cross-layer transcoder (``mntss/clt-gemma-2-2b-426k``).

    python experiments/gemma_clt_prepare.py [--layers 0-25] [--max-minutes 0]

The CLT files stay BF16 on disk and are memory-mapped (DECISIONS.md, D5): 26 encoder files
(75.5 MB each) and 26 decoder files (1.96 GB for layer 0 down to 75.5 MB for layer 25), 28.5 GB
in all; the ``features/`` visualisation files are not needed. Each file is downloaded straight
into the set's directory; finished files are skipped, so the script is resumable. Last, every
layer's tensors are checked against the shapes the twin expects.
"""

from __future__ import annotations

import argparse
import time

from common import ROOT  # noqa: F401  (puts the repo on sys.path)
from huggingface_hub import hf_hub_download

from twinlens.core.gemma2 import Gemma2
from twinlens.twin.transcoders import CLT_REPO, CrossLayerTranscoderSet

N_LAYERS = 26


def _layers(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=f"0-{N_LAYERS - 1}")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--skip-model", action="store_true", help="only the transcoder files")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if not args.skip_model:
        model = Gemma2.load()  # downloads + converts the model on first run
        print(f"model ready: {model.cfg.n_layers} layers, {time.perf_counter() - t0:.0f}s")
    tcs = CrossLayerTranscoderSet(N_LAYERS)
    tcs.dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(CLT_REPO, "config.yaml", local_dir=tcs.dir)
    layers = _layers(args.layers)
    for layer in layers:
        for path in tcs.files(layer):
            if path.exists():
                continue
            if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
                print("time budget reached; rerun to resume", flush=True)
                return
            t = time.perf_counter()
            hf_hub_download(CLT_REPO, path.name, local_dir=tcs.dir)
            mb = path.stat().st_size / 1e6
            print(f"{path.name}: {mb:.0f} MB, {time.perf_counter() - t:.0f}s", flush=True)
    for layer in layers:
        src = tcs[layer]  # checks n_out
        d_model = src.b_dec.shape[0]
        assert src._W_enc.shape == (src.d_sae, d_model), src._W_enc.shape
        assert src._W_dec.shape == (src.d_sae, N_LAYERS - layer, d_model), src._W_dec.shape
        act = "JumpReLU" if src.threshold is not None else "ReLU"
        print(f"L{layer}: d_sae {src.d_sae}, outputs {src.n_out}, {act}, {src._W_dec.dtype}")
    print(f"all layers prepared in {tcs.dir}", flush=True)


if __name__ == "__main__":
    main()
