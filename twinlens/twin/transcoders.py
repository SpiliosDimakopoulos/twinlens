"""Per-layer MLP transcoders for GPT-2 small (Dunefsky, Chlenski & Nanda, 2024).

Convention (verified in DECISIONS.md, D2): each transcoder reads ``LayerNormPre(resid_mid)``,
i.e. the residual stream centred and divided by its LN scale without the affine weights, and
predicts ``hook_mlp_out``:

    acts = relu(x @ W_enc + b_enc)          x = (resid_mid - mean) / scale
    mlp_out ≈ acts @ W_dec + b_dec
"""

from __future__ import annotations

import json
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from twinlens.core.weights import CACHE_DIR, MmapSafetensors, hf_file, npz_to_safetensors

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


# ---------------------------------------------------------------- Qwen3-0.6B
QWEN_REPO = "mwhanna/qwen3-0.6b-transcoders-lowl0"  # circuit-tracer PLTs, 163,840 features/layer

HTTP_ATTEMPTS = 8  # with exponential backoff: ~2 min before giving up
HTTP_TIMEOUT = 60  # seconds
RETRY_CODES = (429, 500, 502, 503, 504)  # rate limit and transient server errors
DOWNLOAD_CHUNK = 32 << 20  # bytes per range request for whole-tensor downloads
ENCODE_CHUNK = 16384  # features per matmul when streaming the BF16 encoder


class RemoteRowTranscoder:
    """Transcoder whose encoder is downloaded once (HTTP range) and whose decoder rows are
    fetched on demand. Lets a 19 GB transcoder set be used with ~335 MB of disk per layer.
    Weights are BF16 on disk and converted to F32 only in the slices that are used."""

    def __init__(self, layer: int, cache_dir: Path | None = None):
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
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(HTTP_ATTEMPTS - 1):
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                    return r.read()
            except urllib.error.HTTPError as e:  # 429: back off as the Hub asks
                if e.code not in RETRY_CODES:
                    raise
                time.sleep(float(e.headers.get("Retry-After") or 2**attempt))
            except OSError:
                time.sleep(2**attempt)
        # Last attempt: let any error propagate.
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read()

    def _header(self):
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

    def _download(self, key: str, dst: Path, chunk: int = DOWNLOAD_CHUNK) -> None:
        a, b = self._span(key)
        tmp = dst.with_suffix(".partial")
        with open(tmp, "wb") as f:
            for s in range(a, b, chunk):
                f.write(self._get(s, min(s + chunk, b)))
        tmp.replace(dst)

    def _small(self, key: str) -> torch.Tensor:
        path = self.dir / f"{key}.bf16"
        if not path.exists():
            self._download(key, path)
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16).copy()).view(torch.bfloat16)

    # -- the interface the twin uses
    def pre(self, x: torch.Tensor, chunk: int = ENCODE_CHUNK) -> torch.Tensor:
        """Full encoder pre-activations, streamed over feature chunks: [..., d_sae]."""
        outs = [x @ self._W_enc[i : i + chunk].float().T for i in range(0, self.d_sae, chunk)]
        return torch.cat(outs, -1) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.pre(x))

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


# ---------------------------------------------------------------- Gemma-2-2B
GEMMA_REPO = "google/gemma-scope-2b-pt-transcoders"  # Gemma Scope, 16,384 features/layer
# average L0 of the 16k transcoders that circuit-tracer uses (mntss/gemma-scope-transcoders)
GEMMA_L0 = (
    *(76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6),
    *(8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41),
)
GEMMA_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")


@dataclass
class JumpReLUTranscoder(Transcoder):
    """``acts = pre * (pre > threshold)``: a feature is active only above its threshold."""

    threshold: torch.Tensor | None = None  # [d_sae]

    def encode(self, x_normed: torch.Tensor) -> torch.Tensor:
        pre = self.pre(x_normed)
        # Gemma Scope thresholds are positive; the clamp keeps "active" meaning "act > 0".
        return pre * (pre > self.threshold.clamp(min=0))


class GemmaScopeTranscoderSet:
    """Gemma Scope JumpReLU transcoders, one per layer, memory-mapped from an F32 safetensors
    copy of each ``params.npz`` (converted on first use).

    Where they read and write was settled on real weights (DECISIONS.md, D4); it stays a parameter:
    ``in_hook`` is ``"hook_mlp_in"`` (pre-feedforward norm output, ``1 + weight`` included) or
    ``"ln2.hook_normalized"`` (without it); ``out_hook`` is ``"hook_mlp_out"`` (after the
    post-feedforward norm) or ``"mlp.hook_out"`` (before it, so that norm stays in the twin).
    ``files`` maps layers to local ``.npz``/``.safetensors`` files instead of the Hub (tests).
    """

    skip_first = False  # Gemma has a BOS token; the sweep never edits it

    def __init__(
        self,
        layers: range | list[int] = range(26),
        in_hook: str = "ln2.hook_normalized",
        out_hook: str = "hook_mlp_out",
        files: dict[int, Path] | None = None,
        cache_dir: Path | None = None,
    ):
        self.layers = list(layers)
        self.in_hook, self.out_hook = in_hook, out_hook
        self.files = files
        self.dir = cache_dir or CACHE_DIR / "twinlens" / "gemma-scope-tc"
        self._cache: dict[int, JumpReLUTranscoder] = {}

    def __contains__(self, layer: int) -> bool:
        return layer in self.layers

    def path(self, layer: int) -> Path:
        """The F32 safetensors file for ``layer`` (downloaded and converted if needed)."""
        if self.files is not None:
            src = Path(self.files[layer])
            if src.suffix == ".safetensors":
                return src
            return npz_to_safetensors(src, src.with_suffix(".f32.safetensors"), GEMMA_KEYS)
        dst = self.dir / f"layer_{layer}.f32.safetensors"
        if dst.exists():
            return dst
        npz = hf_file(
            GEMMA_REPO, f"layer_{layer}/width_16k/average_l0_{GEMMA_L0[layer]}/params.npz"
        )
        return npz_to_safetensors(npz, dst, GEMMA_KEYS)

    def __getitem__(self, layer: int) -> JumpReLUTranscoder:
        if layer not in self.layers:
            raise KeyError(f"no transcoder loaded for layer {layer}")
        if layer not in self._cache:
            st = MmapSafetensors(self.path(layer))
            missing = [k for k in GEMMA_KEYS if k not in st]
            if missing:
                raise KeyError(f"layer {layer}: transcoder file lacks {missing}")
            fields = ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")
            tc = JumpReLUTranscoder(layer, *(st.get(k) for k in fields))
            if tc.W_enc.shape[1] != tc.W_dec.shape[0]:
                raise ValueError(f"layer {layer}: W_enc {tc.W_enc.shape} vs W_dec {tc.W_dec.shape}")
            self._cache[layer] = tc
        return self._cache[layer]


CLT_REPO = "mntss/clt-gemma-2-2b-426k"
ENCODE_ROWS = 4096  # encoder rows converted to F32 at a time (BF16 on disk)


class CrossLayerSource:
    """The features of one source layer ``i`` of a cross-layer transcoder.

    ``acts = relu(resid_mid[i] @ W_enc_iᵀ + b_enc_i)`` (or JumpReLU if the file has a threshold),
    and feature ``f`` writes ``acts[f] * W_dec_i[f, l - i]`` into ``hook_mlp_out[l]`` for every
    ``l >= i``. ``b_dec`` is the output bias of layer ``i`` itself (the circuit-tracer files store
    it next to the encoder), so ``tcs[l].b_dec`` is always the bias of output layer ``l``.
    Weights stay in their stored dtype (BF16 for the released CLTs) on disk; only the rows read
    are converted to F32.
    """

    def __init__(self, layer: int, enc: MmapSafetensors, dec: MmapSafetensors):
        self.layer = layer
        self._W_enc = enc.get(f"W_enc_{layer}")  # [d_sae, d_model], feature-major
        self.b_enc = enc.get(f"b_enc_{layer}").float()
        self.b_dec = enc.get(f"b_dec_{layer}").float()
        key = f"threshold_{layer}"
        self.threshold = enc.get(key).float().reshape(-1) if key in enc else None
        self._W_dec = dec.get(f"W_dec_{layer}")  # [d_sae, n_out, d_model]

    @property
    def d_sae(self) -> int:
        return self._W_enc.shape[0]

    @property
    def n_out(self) -> int:
        return self._W_dec.shape[1]

    def pre(self, x: torch.Tensor, chunk: int = ENCODE_ROWS) -> torch.Tensor:
        parts = [
            x @ self._W_enc[s : s + chunk].float().T + self.b_enc[s : s + chunk]
            for s in range(0, self.d_sae, chunk)
        ]
        return torch.cat(parts, -1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre(x)
        if self.threshold is None:
            return torch.relu(pre)
        return pre * (pre > self.threshold.clamp(min=0))

    def enc_rows(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(idx, dtype=torch.long)
        return self._W_enc[idx].float(), self.b_enc[idx]

    def dec_rows(self, idx) -> torch.Tensor:
        """Decoder rows ``[k, n_out, d_model]``: slot ``j`` writes to layer ``layer + j``."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        if not len(idx):
            return torch.zeros(0, self.n_out, self._W_dec.shape[2])
        # mmap fancy indexing reads only these rows; sorted reads are kinder to the page cache
        order = idx.argsort()
        rows = self._W_dec[idx[order]].float()
        return rows[order.argsort()]


class CrossLayerTranscoderSet:
    """A cross-layer transcoder in circuit-tracer's on-disk format (DECISIONS.md, D5):
    ``W_enc_{i}.safetensors`` (``W_enc_i``, ``b_enc_i``, ``b_dec_i``, optional ``threshold_i``) and
    ``W_dec_{i}.safetensors`` (``W_dec_i`` [d_sae, n_layers - i, d_model]), memory-mapped.

    ``directory`` holds those files (``experiments/gemma_clt_prepare.py`` downloads them); by
    default the project cache. The twin reads ``hook_resid_mid`` (no norm) and writes
    ``hook_mlp_out``; position 0 (BOS) is zeroed, as circuit-tracer's attribution does.
    """

    cross_layer = True
    skip_first = True
    in_hook = "hook_resid_mid"
    out_hook = "hook_mlp_out"

    def __init__(self, n_layers: int = 26, directory: Path | None = None, repo: str = CLT_REPO):
        self.layers = list(range(n_layers))
        self.repo = repo
        self.dir = Path(directory or CACHE_DIR / "twinlens" / repo.replace("/", "--"))
        self._cache: dict[int, CrossLayerSource] = {}

    def __contains__(self, layer: int) -> bool:
        return layer in self.layers

    def files(self, layer: int) -> tuple[Path, Path]:
        return self.dir / f"W_enc_{layer}.safetensors", self.dir / f"W_dec_{layer}.safetensors"

    def __getitem__(self, layer: int) -> CrossLayerSource:
        if layer not in self.layers:
            raise KeyError(f"no cross-layer transcoder source at layer {layer}")
        if layer not in self._cache:
            enc, dec = self.files(layer)
            src = CrossLayerSource(layer, MmapSafetensors(enc), MmapSafetensors(dec))
            if src.n_out != len(self.layers) - layer:
                raise ValueError(f"layer {layer}: W_dec has {src.n_out} output layers")
            self._cache[layer] = src
        return self._cache[layer]
