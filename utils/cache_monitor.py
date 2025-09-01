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
            # L1
            "l1_hit": 0,
            "l1_miss": 0,
            "l1_put": 0,
            "l1_expired": 0,
            "l1_bytes": 0,
            # L2
            "l2_hit": 0,
            "l2_miss": 0,
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
        if nbytes <= 0:
            return
        with self._lock:
            self._counters[key] = int(self._counters.get(key, 0) + int(nbytes))

    # ---- L1 API ----
    def l1_hit(self) -> None: self._incr("l1_hit", 1)
    def l1_miss(self) -> None: self._incr("l1_miss", 1)
    def l1_put(self, nbytes: int) -> None:
        self._incr("l1_put", 1)
        self._add_bytes("l1_bytes", int(nbytes))
    def l1_expired(self) -> None: self._incr("l1_expired", 1)

    # ---- L2 API ----
    def l2_hit(self) -> None: self._incr("l2_hit", 1)
    def l2_miss(self) -> None: self._incr("l2_miss", 1)
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
        l1h, l1m = c.get("l1_hit", 0), c.get("l1_miss", 0)
        l1r = (100.0 * l1h / max(1, (l1h + l1m)))
        l2h, l2m = c.get("l2_hit", 0), c.get("l2_miss", 0)
        l2r = (100.0 * l2h / max(1, (l2h + l2m)))
        l1_bytes = c.get("l1_bytes", 0)
        l2_bytes = c.get("l2_bytes_enq", 0)
        return (
            "CacheMonitor: "
            f"L1(hit={l1h}, miss={l1m}, hit_rate={l1r:.1f}%, puts={c.get('l1_put',0)}, "
            f"expired={c.get('l1_expired',0)}, bytes={self._fmt(l1_bytes)}); "
            f"L2(hit={l2h}, miss={l2m}, hit_rate={l2r:.1f}%, put_enq={c.get('l2_put_enq',0)}, "
            f"bytes_enq={self._fmt(l2_bytes)})"
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

