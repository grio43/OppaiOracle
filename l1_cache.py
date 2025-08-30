from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import threading
import torch

_DTYPE_MAP = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def _canon_dtype(s: str) -> torch.dtype:
    return _DTYPE_MAP.get(str(s).lower(), torch.float32)

def _to_canonical_01(x: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """
    Convert a 0–1 float tensor to the canonical storage dtype.
    For uint8 we quantize to 0..255; other dtypes are straight casts.
    """
    target = _canon_dtype(dtype_str)
    if x.dtype is target:
        return x
    if target is torch.uint8:
        return (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    return x.to(target)

def _from_canonical_01(x: torch.Tensor) -> torch.Tensor:
    """
    Decode canonical storage back to float32 in 0–1 range.
    """
    if x.dtype is torch.uint8:
        return x.to(torch.float32) / 255.0
    return x.to(torch.float32)

@dataclass
class _Entry:
    value: torch.Tensor
    nbytes: int

class ByteLRU:
    """
    Thread-safe LRU bounded by a byte budget. Designed for per-worker usage.
    Stores image tensors (C,H,W) and tiny mask tensors (H,W) by opaque keys (bytes).
    """
    def __init__(self, capacity_bytes: int, dtype: str = "bfloat16"):
        self.capacity = max(0, int(capacity_bytes))
        self.dtype_str = dtype
        self._lock = threading.Lock()
        self._m: "OrderedDict[bytes,_Entry]" = OrderedDict()
        self._size = 0

    @staticmethod
    def _nbytes(t: torch.Tensor) -> int:
        return int(t.numel() * t.element_size())

    def get(self, key: bytes) -> Optional[torch.Tensor]:
        """Return a copy of the stored tensor to avoid aliasing corruption by callers."""
        if self.capacity <= 0:
            return None
        with self._lock:
            e = self._m.get(key)
            if e is None:
                return None
            self._m.move_to_end(key)  # MRU
            # Return a fresh tensor so in-place ops downstream can't corrupt the cache
            # (clone keeps it on CPU and contiguous, matching how values are stored).
            return e.value.detach().clone()

    def put(self, key: bytes, value: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        # store CPU contiguous tensors
        v = value.detach().cpu().contiguous()
        nbytes = self._nbytes(v)
        if nbytes > self.capacity:
            return
        with self._lock:
            old = self._m.pop(key, None)
            if old is not None:
                self._size -= old.nbytes
            self._m[key] = _Entry(v, nbytes)
            self._size += nbytes
            while self._size > self.capacity and self._m:
                _, ev = self._m.popitem(last=False)
                self._size -= ev.nbytes

def build_l1_cache(capacity_mb: int, dtype: str) -> ByteLRU:
    return ByteLRU(capacity_mb * 1024 * 1024, dtype=dtype)

# -- helpers exported for callers (dataset) --
def encode_l1_image_01(x_01: torch.Tensor, *, dtype_str: str) -> torch.Tensor:
    return _to_canonical_01(x_01, dtype_str)

def decode_l1_image_01(x_stored: torch.Tensor) -> torch.Tensor:
    return _from_canonical_01(x_stored)
