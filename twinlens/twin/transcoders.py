"""Per-layer MLP transcoders for GPT-2 small (Dunefsky, Chlenski & Nanda, 2024).

Convention (verified in DECISIONS.md, D2): each transcoder reads ``LayerNormPre(resid_mid)``,
i.e. the residual stream centred and divided by its LN scale without the affine weights, and
predicts ``hook_mlp_out``:

    acts = relu(x @ W_enc + b_enc)          x = (resid_mid - mean) / scale
    mlp_out ≈ acts @ W_dec + b_dec
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from twinlens.core.weights import CACHE_DIR, MmapSafetensors, hf_file

REPO = "jacobdunefsky/gpt2small-transcoders"


@dataclass
class Transcoder:
    layer: int
    W_enc: torch.Tensor  # [d_model, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    W_dec: torch.Tensor  # [d_sae, d_model]
    b_dec: torch.Tensor  # [d_model]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    def pre(self, x_normed: torch.Tensor) -> torch.Tensor:
        return x_normed @ self.W_enc + self.b_enc

    def encode(self, x_normed: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.pre(x_normed))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def enc_rows(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.W_enc[:, idx].T, self.b_enc[idx]

    def dec_rows(self, idx) -> torch.Tensor:
        idx = torch.as_tensor(idx, dtype=torch.long)
        return self.W_dec[idx] if len(idx) else torch.zeros(0, self.W_dec.shape[1])


class TranscoderSet:
    """Lazily memory-maps one transcoder per layer."""

    def __init__(self, n_layers: int = 12):
        self.n_layers = n_layers
        self._cache: dict[int, Transcoder] = {}

    def __getitem__(self, layer: int) -> Transcoder:
        if layer not in self._cache:
            st = MmapSafetensors(hf_file(REPO, f"gpt2-small-dun-chl-mlp-tc{layer}/sae.safetensors"))
            self._cache[layer] = Transcoder(
                layer, st.get("W_enc"), st.get("b_enc"), st.get("W_dec"), st.get("b_dec")
            )
        return self._cache[layer]

    def __iter__(self):
        return (self[layer] for layer in range(self.n_layers))


def layernorm_pre(resid: torch.Tensor, scale: torch.Tensor | None = None, eps: float = 1e-5):
    """Centre and scale like LayerNorm without the affine. ``scale`` can be frozen."""
    x = resid - resid.mean(-1, keepdim=True)
    if scale is None:
        scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return x / scale


# ---------------------------------------------------------------- Qwen3-0.6B
QWEN_REPO = "mwhanna/qwen3-0.6b-transcoders-lowl0"  # circuit-tracer PLTs, 163,840 features/layer


class RemoteRowTranscoder:
    """Transcoder whose encoder is downloaded once (HTTP range) and whose decoder rows are
    fetched on demand. Lets a 19 GB transcoder set be used with ~335 MB of disk per layer.
    Weights are BF16 on disk and converted to F32 only in the slices that are used."""

    def __init__(self, layer: int, cache_dir: Path | None = None):
        import numpy as np

        self.layer = layer
        self.url = f"https://huggingface.co/{QWEN_REPO}/resolve/main/layer_{layer}.safetensors"
        self.dir = (cache_dir or CACHE_DIR / "twinlens" / "qwen-tc") / f"L{layer}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.header, self.data_start = self._header()
        enc = self.dir / "W_enc.bf16"
        if not enc.exists():
            self._download("W_enc", enc)
        self.d_sae, self.d_model = self.header["W_enc"]["shape"]
        raw = np.memmap(enc, dtype=np.uint16, mode="r", shape=(self.d_sae, self.d_model))
        self._W_enc = torch.from_numpy(raw).view(torch.bfloat16)  # [d_sae, d_model]
        self.b_enc = self._small("b_enc").float()
        self.b_dec = self._small("b_dec").float()
        self._dec_cache: dict[int, torch.Tensor] = {}
        cache = self.dir / "dec_rows.pt"
        if cache.exists():
            self._dec_cache = torch.load(cache)

    # -- remote access
    def _get(self, start: int, end: int) -> bytes:
        import time
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except urllib.error.HTTPError as e:  # 429: back off as the Hub asks
                if attempt == 7 or e.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(float(e.headers.get("Retry-After") or 2**attempt))
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _header(self):
        import json
        import struct

        path = self.dir / "header.json"
        if path.exists():
            meta = json.loads(path.read_text())
            return meta["header"], meta["data_start"]
        n = struct.unpack("<Q", self._get(0, 8))[0]
        header = json.loads(self._get(8, 8 + n))
        header.pop("__metadata__", None)
        path.write_text(json.dumps({"header": header, "data_start": 8 + n}))
        return header, 8 + n

    def _span(self, key: str) -> tuple[int, int]:
        a, b = self.header[key]["data_offsets"]
        return self.data_start + a, self.data_start + b

    def _download(self, key: str, dst: Path, chunk: int = 32 << 20) -> None:
        a, b = self._span(key)
        tmp = dst.with_suffix(".partial")
        with open(tmp, "wb") as f:
            for s in range(a, b, chunk):
                f.write(self._get(s, min(s + chunk, b)))
        tmp.replace(dst)

    def _small(self, key: str) -> torch.Tensor:
        import numpy as np

        path = self.dir / f"{key}.bf16"
        if not path.exists():
            self._download(key, path)
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16).copy()).view(torch.bfloat16)

    # -- the interface the twin uses
    def pre(self, x: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """Full encoder pre-activations, streamed over feature chunks: [..., d_sae]."""
        outs = [x @ self._W_enc[i : i + chunk].float().T for i in range(0, self.d_sae, chunk)]
        return torch.cat(outs, -1) + self.b_enc

    def enc_rows(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._W_enc[idx].float(), self.b_enc[idx]

    def prefetch_decoder(self) -> None:
        """Download the full decoder (one sequential range request, ~335 MB) so row lookups
        are local. Call ``drop_decoder`` afterwards; the rows used stay cached."""
        path = self.dir / "W_dec.bf16"
        if not path.exists():
            self._download("W_dec", path)

    def drop_decoder(self) -> None:
        self._save_cache()
        (self.dir / "W_dec.bf16").unlink(missing_ok=True)

    def _save_cache(self) -> None:
        torch.save(self._dec_cache, self.dir / "dec_rows.pt")

    def dec_rows(self, idx: torch.Tensor | list[int]) -> torch.Tensor:
        import numpy as np

        ids = [int(i) for i in (idx.tolist() if isinstance(idx, torch.Tensor) else idx)]
        missing = sorted({i for i in ids if i not in self._dec_cache})
        if missing:
            full = self.dir / "W_dec.bf16"
            if full.exists():
                raw = np.memmap(full, dtype=np.uint16, mode="r", shape=(self.d_sae, self.d_model))
                rows = (
                    torch.from_numpy(np.ascontiguousarray(raw[missing]))
                    .view(torch.bfloat16)
                    .float()
                )
                for i, r in zip(missing, rows, strict=True):
                    self._dec_cache[i] = r
            else:  # a few rows: individual range requests (rate-limited by the Hub)
                a, _ = self._span("W_dec")
                row = 2 * self.d_model
                for i in missing:
                    buf = bytearray(self._get(a + i * row, a + (i + 1) * row))
                    self._dec_cache[i] = torch.frombuffer(buf, dtype=torch.bfloat16).float().clone()
                self._save_cache()
        if not ids:
            return torch.zeros(0, self.d_model)
        return torch.stack([self._dec_cache[i] for i in ids])


class QwenTranscoderSet:
    """Transcoders for a subset of Qwen3 layers (others keep their real MLP)."""

    def __init__(self, layers: range = range(10, 28)):
        self.layers = list(layers)
        self._cache: dict[int, RemoteRowTranscoder] = {}

    def __contains__(self, layer: int) -> bool:
        return layer in self.layers

    def __getitem__(self, layer: int) -> RemoteRowTranscoder:
        if layer not in self.layers:
            raise KeyError(f"no transcoder loaded for layer {layer}")
        if layer not in self._cache:
            self._cache[layer] = RemoteRowTranscoder(layer)
        return self._cache[layer]
