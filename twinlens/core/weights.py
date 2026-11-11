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
from collections.abc import Iterable
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

_ROOT = Path(__file__).resolve().parents[2]
# A pip install puts the package in site-packages (possibly read-only, and a bad home for 2.3 GB),
# so only a source checkout keeps the repo-root cache that CI caches.
CACHE_DIR = Path(
    os.environ.get("TWINLENS_CACHE")
    or (
        _ROOT / ".hf_cache"
        if (_ROOT / "pyproject.toml").exists()
        else Path.home() / ".cache" / "twinlens"
    )
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


def npz_to_safetensors(
    src: str | Path, dst: str | Path, keys: Iterable[str] | None = None, chunk: int = 16 << 20
) -> Path:
    """Rewrite an ``.npz`` archive as an F32 safetensors file, streaming each array in
    ``chunk``-byte pieces so peak RAM stays small. ``keys`` (if given) must all be present."""
    import zipfile

    from numpy.lib import format as npf

    dst = Path(dst)
    if dst.exists():
        return dst
    with zipfile.ZipFile(src) as z:
        entries = {}
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with z.open(info) as f:
                version = npf.read_magic(f)
                read_header = (
                    npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
                )
                shape, fortran, dtype = read_header(f)
            if fortran and len(shape) > 1:
                raise ValueError(f"{src}: {info.filename} is Fortran-ordered")
            entries[info.filename[:-4]] = (info, tuple(shape), np.dtype(dtype))
        if keys is not None:
            missing = sorted(set(keys) - set(entries))
            if missing:
                raise KeyError(f"{src}: missing arrays {missing}; found {sorted(entries)}")
        header, off = {}, 0
        for k, (_, shape, _) in sorted(entries.items()):
            n = 4 * int(np.prod(shape))
            header[k] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n]}
            off += n
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".partial")
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for _, (info, _, dtype) in sorted(entries.items()):
                with z.open(info) as f:
                    version = npf.read_magic(f)  # skip the .npy header
                    (npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0)(
                        f
                    )
                    step = chunk - chunk % dtype.itemsize
                    while buf := f.read(step):
                        out.write(np.frombuffer(buf, dtype=dtype).astype("<f4").tobytes())
        tmp.replace(dst)
    return dst
