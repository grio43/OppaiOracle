from __future__ import annotations
import os, time, atexit, signal, queue, logging, threading
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

# Thread-local storage for batch read transactions
_thread_local = threading.local()


class _BatchReadContext:
    """Context manager for batched LMDB reads using a single transaction."""

    def __init__(self, reader: "LMDBReader"):
        self.reader = reader
        self.txn = None

    def __enter__(self):
        # Start a read transaction and store it in thread-local storage
        self.txn = self.reader.env.begin(write=False)
        if not hasattr(_thread_local, 'lmdb_txns'):
            _thread_local.lmdb_txns = []
        _thread_local.lmdb_txns.append(self.txn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the transaction and remove from thread-local storage
        if hasattr(_thread_local, 'lmdb_txns') and _thread_local.lmdb_txns:
            _thread_local.lmdb_txns.pop()
        if self.txn is not None:
            try:
                self.txn.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
        return False


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
        """Get a value from LMDB cache.

        If called within a batch_context(), uses the shared transaction for
        better performance. Otherwise creates a short-lived transaction.
        """
        # Check if we're in a batch context with an active transaction
        if hasattr(_thread_local, 'lmdb_txns') and _thread_local.lmdb_txns:
            # Use the current batch transaction
            txn = _thread_local.lmdb_txns[-1]
            v = txn.get(key)
            if v is None:
                monitor.l2_miss()
            else:
                monitor.l2_hit()
            return v
        else:
            # Create short-lived read transaction (original behavior)
            with self.env.begin(write=False) as txn:
                v = txn.get(key)
                if v is None:
                    monitor.l2_miss()
                else:
                    monitor.l2_hit()
                return v

    def get_many(self, keys: list[bytes]) -> list[Optional[bytes]]:
        """Get multiple keys in a single transaction for better performance.

        This reduces transaction overhead from 50-100μs per key to amortized
        cost across all keys in the batch.

        Args:
            keys: List of keys to retrieve

        Returns:
            List of values (or None for missing keys) in the same order as keys
        """
        if not keys:
            return []

        # Single transaction for all gets
        results = []
        with self.env.begin(write=False) as txn:
            for key in keys:
                v = txn.get(key)
                results.append(v)
                if v is None:
                    monitor.l2_miss()
                else:
                    monitor.l2_hit()
        return results

    def batch_context(self):
        """Context manager for keeping a transaction open across multiple gets.

        Use this when you need to do multiple get() calls and want to amortize
        transaction overhead. The transaction is shared via a thread-local variable.

        ⚠️ WARNING: Do not hold this context for extended periods!
        - Long-lived read transactions block LMDB writer from reclaiming pages
        - Can cause MapFull errors if writer runs out of space
        - Keep context lifetime under 100ms for safety
        - For batch operations, prefer get_many() instead

        Example:
            # GOOD: Short-lived context
            with reader.batch_context():
                val1 = reader.get(key1)
                val2 = reader.get(key2)

            # BETTER: Use get_many for batch reads
            vals = reader.get_many([key1, key2])

            # BAD: Long-lived context blocks writer!
            # with reader.batch_context():
            #     for i in range(10000):
            #         val = reader.get(keys[i])  # Don't do this!
        """
        return _BatchReadContext(self)

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
        max_map_size_multiplier: int = 2  # Maximum growth factor (2x = 17 TB max)
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
        self._flush_lock = threading.Lock()  # Thread-safe flush guard (CR-012)

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

            # Thread-safe guard against re-entry (CR-012)
            if not self._flush_lock.acquire(blocking=False):
                logging.debug("Flush already in progress, skipping")
                return

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
                self._flush_lock.release()  # CR-012: Release lock properly

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

        # Final flush - flush_pending() will handle the lock itself
        try:
            flush_pending()
            env.close()
        except Exception as e:
            logging.warning(f"L2 cache writer final flush failed: {e}")

def start_l2_writer(path: str, map_size_bytes: int, max_map_size_multiplier: int = 2) -> Tuple[mp.Queue, mp.Process]:
    """
    Starts and returns a dedicated L2 cache writer process and its command queue.

    Args:
        path: The directory path for the LMDB environment.
        map_size_bytes: The initial size of the LMDB memory map.
        max_map_size_multiplier: Maximum growth factor for map size (default: 2x).
            For example, if map_size_bytes=8.5TB and multiplier=2, max is 17TB.

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
            q.put_nowait(None)  # Send poison pill to trigger graceful shutdown
        except Exception:
            pass
        try:
            # Adaptive timeout based on queue size
            # Estimate: ~10ms per item for flush, with min 5s and max 30s
            queue_size = q.qsize()
            adaptive_timeout = max(5.0, min(30.0, queue_size * 0.01))

            logging.info(f"L2 cache writer shutting down (queue size: {queue_size}, timeout: {adaptive_timeout:.1f}s)...")
            proc.join(timeout=adaptive_timeout)

            if proc.is_alive():
                logging.warning(
                    f"L2 cache writer did not finish in {adaptive_timeout:.1f}s "
                    f"(queue had {queue_size} pending writes). Some writes may be lost. "
                    f"Consider increasing the timeout or reducing write frequency."
                )
            else:
                logging.info("L2 cache writer shutdown complete")
        except Exception as e:
            logging.warning(f"Error during L2 writer shutdown: {e}")

    atexit.register(_stop)
    return q, proc
