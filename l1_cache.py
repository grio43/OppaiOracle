from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, TypedDict
import threading
import time
import torch

# Try to import monitor, fall back to no-op if unavailable
try:
    from utils.cache_monitor import monitor
except ImportError:
    # Fallback: create a no-op monitor for when utils.cache_monitor is unavailable
    class _NoOpMonitor:
        """No-op monitor for when utils.cache_monitor is unavailable."""
        def l1_hit(self) -> None:
            pass
        def l1_miss(self) -> None:
            pass
        def l1_put(self, nbytes: int) -> None:
            pass
        def l1_evict(self, nbytes: int) -> None:
            pass
    monitor = _NoOpMonitor()

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
    Decode canonical storage back to 0–1 range, preserving dtype for non-uint8.
    - uint8 is dequantized to float32 in [0,1].
    - float16/bfloat16/float32 are returned as-is to honor configured precision.
    """
    if x.dtype is torch.uint8:
        return x.to(torch.float32) / 255.0
    return x

@dataclass
class _Entry:
    value: torch.Tensor
    nbytes: int

# CR-044: TypedDict for cache_info return type
class CacheInfo(TypedDict):
    """Structure returned by ByteLRU.cache_info().

    Fields:
        capacity_bytes: Maximum cache size in bytes
        size_bytes: Current cache size in bytes
        items: Number of items currently in cache
        hits: Number of cache hits (None if track_stats=False)
        misses: Number of cache misses (None if track_stats=False)
        expired: Number of expired entries (None if track_stats=False)
    """
    capacity_bytes: int
    size_bytes: int
    items: int
    hits: Optional[int]
    misses: Optional[int]
    expired: Optional[int]

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
                now = time.monotonic()  # monotonic clock
                if exp is not None and now >= exp:
                    entry = self._m.pop(key, None)
                    if entry is not None:
                        self._size -= entry.nbytes
                    self._expires.pop(key, None)
                    if self.track_stats:
                        self._expired += 1
                        self._misses += 1
                    # Monitor: treat expiry as a miss and expired event
                    monitor.l1_expired()
                    monitor.l1_miss()
                    return None
            e = self._m.get(key)
            if e is None:
                if self.track_stats:
                    self._misses += 1
                monitor.l1_miss()
                return None
            # update MRU order
            self._m.move_to_end(key)
            # update sliding TTL expiry
            if self.ttl is not None:
                self._expires[key] = time.monotonic() + self.ttl
            if self.track_stats:
                self._hits += 1
            monitor.l1_hit()
            return e.value.detach().clone()

    def put(self, key: bytes, value: torch.Tensor) -> None:
        """Store a tensor in the cache.

        Args:
            key: Unique identifier (must be bytes)
            value: Tensor to cache (any device, any dtype)

        Note:
            - Input tensor is automatically detached, moved to CPU, and converted to canonical dtype
            - If tensor is larger than cache capacity, it will not be stored
            - LRU eviction occurs automatically when capacity is exceeded
            - Thread-safe operation

        Raises:
            TypeError: If key is not bytes or value is not a torch.Tensor
        """
        if self.capacity <= 0:
            return

        # Input validation
        if not isinstance(key, bytes):
            raise TypeError(
                f"key must be of type bytes, got {type(key).__name__}. "
                f"Convert your key to bytes before calling put()."
            )

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"value must be a torch.Tensor, got {type(value).__name__}. "
                f"Ensure you're passing a PyTorch tensor to the cache."
            )

        # convert to canonical dtype on CPU for accurate byte accounting
        v = value.detach().cpu().contiguous().to(_canon_dtype(self.dtype_str))
        nbytes = self._nbytes(v)
        if nbytes > self.capacity:
            return
        with self._lock:
            old = self._m.pop(key, None)
            if old is not None:
                self._size -= old.nbytes
            self._m[key] = _Entry(v, nbytes)
            if self.ttl is not None:
                self._expires[key] = time.monotonic() + self.ttl
            self._size += nbytes
            monitor.l1_put(nbytes)
            while self._size > self.capacity and self._m:
                k_old, ev = self._m.popitem(last=False)
                self._size -= ev.nbytes
                if self.ttl is not None:
                    self._expires.pop(k_old, None)

    def cache_info(self) -> CacheInfo:
        """Return a snapshot of cache statistics (hits, misses, expired).

        Returns:
            Dictionary with cache statistics. See CacheInfo for field descriptions.
        """
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
    track_stats: Optional[bool] = None,
) -> ByteLRU:
    """Create a ByteLRU with the given size and optional TTL/stats flags.

    If track_stats is None, it follows the global cache monitor enable flag.
    """
    return ByteLRU(
        capacity_mb * 1024 * 1024,
        dtype=dtype,
        ttl_seconds=ttl_seconds,
        track_stats=(monitor.enabled if track_stats is None else bool(track_stats)),
    )

# -- helpers exported for callers (dataset) --
def encode_l1_image_01(x_01: torch.Tensor, *, dtype_str: str) -> torch.Tensor:
    return _to_canonical_01(x_01, dtype_str)

def decode_l1_image_01(x_stored: torch.Tensor) -> torch.Tensor:
    return _from_canonical_01(x_stored)
