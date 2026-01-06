import os
import atexit
import threading
import time
import logging
from typing import Dict, Optional


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    v = str(val).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


class CacheMonitor:
    """
    Ultra-lightweight, process-local cache monitor.

    - Enabled via env var `OO_CACHE_MONITOR` (default: disabled)
    - Emits a one-line summary at process exit when enabled
    - Thread-safe counters; no I/O on the hot path
    """

    def __init__(self, enabled: Optional[bool] = None) -> None:
        if enabled is None:
            enabled = _truthy(os.getenv("OO_CACHE_MONITOR"))
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            # Sidecar cache metrics (l2_ prefix kept for backward compatibility)
            "l2_hit": 0,
            "l2_miss": 0,
            "l2_stale": 0,  # Cache entries that loaded but failed validation (shape/dtype/hash mismatch)
            "l2_put_enq": 0,
            "l2_bytes_enq": 0,
        }
        # Avoid import-time side effects when disabled
        if self.enabled and _truthy(os.getenv("OO_CACHE_MONITOR_AUTOPRINT", "1")):
            atexit.register(self.log_summary)

    # ---- internal helpers ----
    def _incr(self, key: str, delta: int = 1) -> None:
        if not self.enabled:
            return
        if delta == 0:
            return
        with self._lock:
            self._counters[key] = int(self._counters.get(key, 0) + int(delta))

    def _add_bytes(self, key: str, nbytes: int) -> None:
        if not self.enabled:
            return
        if nbytes < 0:
            # Log warning for negative bytes - indicates bug in caller
            logging.getLogger("cache_monitor").warning(
                f"Negative byte count {nbytes} for {key} - this indicates a bug in the caller"
            )
            return
        if nbytes == 0:
            return
        with self._lock:
            self._counters[key] = int(self._counters.get(key, 0) + int(nbytes))

    # ---- Sidecar Cache API (l2_ method names kept for backward compatibility) ----
    def l2_hit(self) -> None: self._incr("l2_hit", 1)
    def l2_miss(self) -> None: self._incr("l2_miss", 1)
    def l2_stale(self) -> None: self._incr("l2_stale", 1)  # Cache loaded but failed validation
    def l2_put_enqueued(self, nbytes: int) -> None:
        self._incr("l2_put_enq", 1)
        self._add_bytes("l2_bytes_enq", int(nbytes))

    # ---- Reporting ----
    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def _fmt(self, n: int) -> str:
        # human bytes
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
            n /= 1024.0
        return f"{n:.1f}TB"

    def format_summary(self) -> str:
        c = self.snapshot()
        hits, misses, stale = c.get("l2_hit", 0), c.get("l2_miss", 0), c.get("l2_stale", 0)
        # Hit rate excludes stale (which were loaded but invalid)
        total_requests = hits + misses + stale
        hit_rate = (100.0 * hits / max(1, total_requests))
        bytes_enq = c.get("l2_bytes_enq", 0)
        return (
            "CacheMonitor: "
            f"Sidecar(hit={hits}, miss={misses}, stale={stale}, hit_rate={hit_rate:.1f}%, "
            f"put_enq={c.get('l2_put_enq',0)}, bytes_enq={self._fmt(bytes_enq)})"
        )

    def log_summary(self) -> None:
        if not self.enabled:
            return
        msg = self.format_summary()
        try:
            logging.getLogger("cache_monitor").info(msg)
        except Exception:
            # Last-resort print to stdout
            print(msg)


# Public singleton
monitor = CacheMonitor()

