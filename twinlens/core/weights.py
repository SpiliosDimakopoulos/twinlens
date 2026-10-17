"""Zero-copy access to safetensors files.

Tensors are returned as read-only views over a memory-mapped file. File-backed pages are served
from the OS page cache and do not count towards the process' committed memory, so large weight
files (model + transcoders) can be used on machines with very little free RAM.
"""

from __future__ import annotations

import json
import os
import struct
import warnings
from pathlib import Path

import numpy as np
import torch

_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "F64": np.float64,
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

CACHE_DIR = Path(
    os.environ.get("TWINLENS_CACHE", Path(__file__).resolve().parents[2] / ".hf_cache")
)


class MmapSafetensors:
    """Read-only, lazily-mapped view of a ``.safetensors`` file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        self.metadata = header.pop("__metadata__", {})
        self._entries = header
        self._data_offset = 8 + header_len

    def keys(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._entries[key]["shape"])

    def get(self, key: str) -> torch.Tensor:
        entry = self._entries[key]
        dtype = entry["dtype"]
        if dtype not in _DTYPES and dtype != "BF16":
            raise TypeError(f"{key}: unsupported dtype {dtype} for mmap")
        start, _ = entry["data_offsets"]
        shape = tuple(entry["shape"])
        arr = np.memmap(
            self.path,
            dtype=np.uint16 if dtype == "BF16" else _DTYPES[dtype],
            mode="r",
            offset=self._data_offset + start,
            shape=shape if shape else (1,),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # non-writable numpy array
            t = torch.from_numpy(arr)
        if dtype == "BF16":  # numpy has no bfloat16: reinterpret the raw 16-bit words
            t = t.view(torch.bfloat16)
        return t if shape else t.reshape(())


def hf_file(repo_id: str, filename: str, revision: str | None = None) -> Path:
    """Download (or reuse) one file from the Hugging Face Hub into the project cache."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, revision=revision, cache_dir=CACHE_DIR / "hub"))


def convert_to_f32(src: str | Path, dst: str | Path, chunk_rows: int = 4096) -> Path:
    """Rewrite a (BF16/F16) safetensors file as F32, streaming so peak RAM stays small.
    CPUs without native bf16 run F32 matmuls much faster, and an mmapped F32 file costs page
    cache, not committed memory."""
    dst = Path(dst)
    if dst.exists():
        return dst
    st = MmapSafetensors(src)
    keys = sorted(st.keys())
    header, off = {}, 0
    for k in keys:
        n = int(np.prod(st.shape(k))) if st.shape(k) else 1
        header[k] = {"dtype": "F32", "shape": list(st.shape(k)), "data_offsets": [off, off + 4 * n]}
        off += 4 * n
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for k in keys:
            t = st.get(k)
            flat = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
            for i in range(0, flat.shape[0], chunk_rows):
                f.write(flat[i : i + chunk_rows].float().contiguous().numpy().tobytes())
    tmp.replace(dst)
    return dst
