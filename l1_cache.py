from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict
import threading
import time
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

    Enhancements:
      - Optional TTL (time‑to‑live) support. When ttl_seconds is set to a positive
        number, each entry expires after that many seconds since last access. If an
        entry is expired it will be removed on the next lookup.
      - Optional statistics tracking. When track_stats=True the cache counts hits,
        misses and expirations. Call cache_info() to inspect stats.
    """
    def __init__(
        self,
        capacity_bytes: int,
        dtype: str = "bfloat16",
        ttl_seconds: Optional[float] = None,
        track_stats: bool = False,
    ):
        self.capacity = max(0, int(capacity_bytes))
        self.dtype_str = dtype
        # TTL: expire entries after ttl_seconds since last access (None disables TTL)
        self.ttl: Optional[float] = float(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else None
        self.track_stats = bool(track_stats)
        self._lock = threading.Lock()
        self._m: "OrderedDict[bytes,_Entry]" = OrderedDict()
        # per‑key expiry timestamps when TTL is enabled
        self._expires: Dict[bytes, float] = {}
        self._size = 0
        if self.track_stats:
            self._hits = 0
            self._misses = 0
            self._expired = 0

    @staticmethod
    def _nbytes(t: torch.Tensor) -> int:
        return int(t.numel() * t.element_size())

    def get(self, key: bytes) -> Optional[torch.Tensor]:
        """Return a copy of the stored tensor; handle TTL expiry and stats."""
        if self.capacity <= 0:
            return None
        with self._lock:
            # If TTL enabled, purge expired entry on access
            if self.ttl is not None:
                exp = self._expires.get(key)
                if exp is not None and time.time() >= exp:
                    entry = self._m.pop(key, None)
                    if entry is not None:
                        self._size -= entry.nbytes
                    self._expires.pop(key, None)
                    if self.track_stats:
                        self._expired += 1
                        self._misses += 1
                    return None
            e = self._m.get(key)
            if e is None:
                if self.track_stats:
                    self._misses += 1
                return None
            # update MRU order
            self._m.move_to_end(key)
            # update sliding TTL expiry
            if self.ttl is not None:
                self._expires[key] = time.time() + self.ttl
            if self.track_stats:
                self._hits += 1
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
            if self.ttl is not None:
                self._expires[key] = time.time() + self.ttl
            self._size += nbytes
            while self._size > self.capacity and self._m:
                k_old, ev = self._m.popitem(last=False)
                self._size -= ev.nbytes
                if self.ttl is not None:
                    self._expires.pop(k_old, None)

    def cache_info(self) -> dict:
        """Return a snapshot of cache statistics (hits, misses, expired)."""
        with self._lock:
            return {
                "capacity_bytes": self.capacity,
                "size_bytes": self._size,
                "items": len(self._m),
                "hits": getattr(self, "_hits", None),
                "misses": getattr(self, "_misses", None),
                "expired": getattr(self, "_expired", None),
            }

def build_l1_cache(
    capacity_mb: int,
    dtype: str,
    ttl_seconds: Optional[float] = None,
    track_stats: bool = False,
) -> ByteLRU:
    """Create a ByteLRU with the given size and optional TTL/stats flags."""
    return ByteLRU(
        capacity_mb * 1024 * 1024,
        dtype=dtype,
        ttl_seconds=ttl_seconds,
        track_stats=track_stats,
    )

# -- helpers exported for callers (dataset) --
def encode_l1_image_01(x_01: torch.Tensor, *, dtype_str: str) -> torch.Tensor:
    return _to_canonical_01(x_01, dtype_str)

def decode_l1_image_01(x_stored: torch.Tensor) -> torch.Tensor:
    return _from_canonical_01(x_stored)
