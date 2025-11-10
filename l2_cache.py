from __future__ import annotations
import os, time, atexit, signal, queue, logging
import multiprocessing as mp
from typing import Optional, Tuple
try:
    import lmdb  # type: ignore
except ModuleNotFoundError:
    lmdb = None  # optional dependency
import torch
from cache_codec import encode_tensor as _tensor_to_bytes, decode_tensor as _tensor_from_bytes
from utils.cache_monitor import monitor

# Small helper to reduce readonly-open races before the writer has created files.
def _wait_for_env_files(path: str, *, timeout_s: float = 2.0, poll_ms: int = 50) -> None:
    import os, time
    end = time.time() + float(timeout_s)
    data = os.path.join(path, "data.mdb")
    while not os.path.exists(data) and time.time() < end:
        time.sleep(poll_ms / 1000.0)

# Serialization now lives in cache_codec.py (safetensors + optional HMAC)

# --- Read-only per-process cache handle ---
class LMDBReader:
    """
    A read-only handle to an LMDB environment.

    This class is a context manager, ensuring that the environment is
    properly closed when the context is exited.
    """
    def __init__(self, path: str, map_size_bytes: int, max_readers: int = 4096):
        if lmdb is None:
            raise RuntimeError(
                "LMDB is not installed but L2 cache was requested. "
                "Install with `pip install lmdb` or disable L2 cache."
            )
        # Retry readonly open with backoff to avoid races with writer bootstrapping
        attempts = 3
        env = None
        last_error: Optional[Exception] = None

        for i in range(attempts):
            try:
                _wait_for_env_files(path)
                env = lmdb.open(
                    path,
                    map_size=map_size_bytes,
                    subdir=True,
                    readonly=True,
                    lock=True,
                    readahead=False,
                    max_readers=max_readers,
                    max_dbs=1,
                )
                break
            except lmdb.Error as e:
                last_error = e
                if i < attempts - 1:
                    time.sleep(0.05)
            except Exception as e:
                # Catch non-LMDB errors (OSError, PermissionError, etc.)
                last_error = e
                if i < attempts - 1:
                    time.sleep(0.05)

        # Ensure env was successfully opened
        if env is None:
            raise RuntimeError(
                f"Failed to open LMDB environment at {path} after {attempts} attempts. "
                f"Last error: {last_error}. "
                f"Check that the path exists, is readable, and is a valid LMDB database."
            )

        self.env: lmdb.Environment = env

    def get(self, key: bytes) -> Optional[bytes]:
        # short-lived read txns so the writer never waits on us
        with self.env.begin(write=False) as txn:
            v = txn.get(key)
            if v is None:
                monitor.l2_miss()
            else:
                monitor.l2_hit()
            return v

    def close(self) -> None:
        try:
            if self.env:
                self.env.close()
                self.env = None
        except Exception:
            pass

    def __enter__(self) -> LMDBReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

# --- Dedicated writer process (single writer, batched commits) ---
class _WriterProc(mp.Process):
    """
    A dedicated writer process for LMDB.

    This process listens on a queue for (key, value) pairs and writes
    them to the database in batches to improve performance. It handles
    database map resizing and graceful shutdown.
    """
    def __init__(
        self,
        path: str,
        map_size_bytes: int,
        q: mp.Queue,
        batch_items: int = 512,
        batch_bytes: int = 64 << 20,   # 64MB
        flush_ms: int = 50,
        max_map_size_multiplier: int = 10  # Maximum growth factor
    ):
        super().__init__(daemon=True)
        self.path = path
        self.map_size_bytes = map_size_bytes
        self.max_map_size = map_size_bytes * max_map_size_multiplier
        self.q = q
        self.batch_items = batch_items
        self.batch_bytes = batch_bytes
        self.flush_s = flush_ms / 1000.0
        self._running = mp.Event()
        self._running.set()
        self._in_flush = False  # Guard against re-entrant flush calls

        # Cache statistics tracking (CR-006)
        self.stats = {
            'writes': 0,
            'dropped_batches': 0,
            'dropped_items': 0,
            'cache_full_events': 0,
        }
        self.cache_full = False  # Backpressure signal

    def run(self) -> None:
        env = lmdb.open(
            self.path,
            map_size=self.map_size_bytes,
            subdir=True,
            readonly=False,
            lock=True,
            writemap=True,
            map_async=True,   # async msync on background thread
            sync=False,       # don't fsync every commit
            readahead=False,
            max_dbs=1
        )

        pending = []
        pending_bytes = 0
        last_flush = time.monotonic()

        def flush_pending():
            nonlocal pending, pending_bytes, last_flush
            if not pending:
                return

            # Guard against re-entry (e.g., from signal handler)
            if self._in_flush:
                logging.warning("flush_pending called re-entrantly, skipping")
                return

            self._in_flush = True
            try:
                with env.begin(write=True) as txn:
                    for k, v in pending:
                        txn.put(k, v, overwrite=True)
            except lmdb.MapFullError:
                info = env.info()
                current_size = info.get("map_size", self.map_size_bytes)
                new_size = int(max(self.map_size_bytes, current_size) * 2)

                # Check against maximum
                if new_size > self.max_map_size:
                    logging.error(
                        f"L2 cache map size would exceed maximum: "
                        f"current={current_size / (1024**3):.2f}GB, "
                        f"requested={new_size / (1024**3):.2f}GB, "
                        f"max={self.max_map_size / (1024**3):.2f}GB. "
                        f"Consider increasing max_map_size or reducing cache usage."
                    )
                    # Try setting to max instead of failing entirely
                    new_size = self.max_map_size

                    # Last chance: try with max size
                    try:
                        env.set_mapsize(new_size)
                        with env.begin(write=True) as txn:
                            for k, v in pending:
                                txn.put(k, v, overwrite=True)
                    except lmdb.MapFullError:
                        # Can't grow anymore, cache is truly full (CR-006)
                        self.cache_full = True
                        self.stats['cache_full_events'] += 1
                        self.stats['dropped_batches'] += 1
                        self.stats['dropped_items'] += len(pending)

                        logging.critical(
                            f"L2 cache FULL: Dropped batch of {len(pending)} items. "
                            f"Total dropped: {self.stats['dropped_items']} items in "
                            f"{self.stats['dropped_batches']} batches. "
                            f"Cache size: {new_size / (1024**3):.2f}GB. "
                            f"Increase max_map_size or clear the cache."
                        )
                        # Drop the batch to avoid blocking
                        return

                else:
                    # Log resize warnings for significant growth
                    growth_factor = new_size / self.map_size_bytes
                    if growth_factor >= 4:
                        logging.warning(
                            f"L2 cache map size growing significantly: "
                            f"{current_size / (1024**3):.2f}GB → {new_size / (1024**3):.2f}GB "
                            f"({growth_factor:.1f}x initial size). "
                            f"This may indicate excessive cache usage."
                        )

                    env.set_mapsize(new_size)
                    with env.begin(write=True) as txn:
                        for k, v in pending:
                            txn.put(k, v, overwrite=True)

            finally:
                pending.clear()
                pending_bytes = 0
                last_flush = time.monotonic()
                self._in_flush = False

        def _shutdown(*_):
            """Signal handler: just set flag, don't do work."""
            self._running.clear()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        while self._running.is_set():
            timeout = max(0.0, self.flush_s - (time.monotonic() - last_flush))
            try:
                item = self.q.get(timeout=timeout)
                if item is None:  # poison pill
                    self._running.clear()
                    break
                k, v = item
                pending.append((k, v))
                pending_bytes += len(v)
            except queue.Empty:
                pass

            if (
                pending
                and (len(pending) >= self.batch_items
                     or pending_bytes >= self.batch_bytes
                     or (time.monotonic() - last_flush) >= self.flush_s)
            ):
                flush_pending()

        # Final flush (only if not already flushing from signal)
        if not self._in_flush:
            try:
                flush_pending()
                env.close()
            except Exception as e:
                logging.warning(f"L2 cache writer final flush failed: {e}")

def start_l2_writer(path: str, map_size_bytes: int, max_map_size_multiplier: int = 10) -> Tuple[mp.Queue, mp.Process]:
    """
    Starts and returns a dedicated L2 cache writer process and its command queue.

    Args:
        path: The directory path for the LMDB environment.
        map_size_bytes: The initial size of the LMDB memory map.
        max_map_size_multiplier: Maximum growth factor for map size (default: 10x).
            For example, if map_size_bytes=100GB and multiplier=10, max is 1TB.

    Returns:
        A tuple containing the multiprocessing queue for commands and the
        handle to the running writer process.
    """
    if lmdb is None:
        raise RuntimeError("LMDB is required to start L2 writer. Install `lmdb`.")

    # The writer process is responsible for creating the cache directory.
    os.makedirs(path, exist_ok=True)

    ctx = mp.get_context("spawn")
    import os as _os
    q_max = int(_os.environ.get("L2_WRITER_QUEUE_MAX", "1024"))
    # Keep a floor to avoid pathological tiny queues.
    q: mp.Queue = ctx.Queue(maxsize=max(64, q_max))
    proc = _WriterProc(path=path, map_size_bytes=map_size_bytes, q=q, max_map_size_multiplier=max_map_size_multiplier)
    proc.start()

    def _stop():
        try:
            q.put_nowait(None)
        except Exception:
            pass
        try:
            proc.join(timeout=2.0)
        except Exception:
            pass

    atexit.register(_stop)
    return q, proc
