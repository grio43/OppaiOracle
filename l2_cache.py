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
            except lmdb.Error:
                if i < attempts - 1:
                    time.sleep(0.05)
                else:
                    raise
        self.env = env

    def get(self, key: bytes) -> Optional[bytes]:
        # short-lived read txns so the writer never waits on us
        with self.env.begin(write=False) as txn:
            return txn.get(key)

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
        flush_ms: int = 50
    ):
        super().__init__(daemon=True)
        self.path = path
        self.map_size_bytes = map_size_bytes
        self.q = q
        self.batch_items = batch_items
        self.batch_bytes = batch_bytes
        self.flush_s = flush_ms / 1000.0
        self._running = mp.Event()
        self._running.set()

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
            try:
                with env.begin(write=True) as txn:
                    for k, v in pending:
                        txn.put(k, v, overwrite=True)
            except lmdb.MapFullError:
                info = env.info()
                new_size = int(max(self.map_size_bytes, info.get("map_size", self.map_size_bytes)) * 2)
                env.set_mapsize(new_size)
                with env.begin(write=True) as txn:
                    for k, v in pending:
                        txn.put(k, v, overwrite=True)
            pending.clear()
            pending_bytes = 0
            last_flush = time.monotonic()

        def _shutdown(*_):
            # flush pending writes and stop
            self._running.clear()
            try:
                flush_pending()
            except Exception as e:
                logging.warning(f"L2 cache writer shutdown flush failed: {e}")

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

        # Final flush
        try:
            flush_pending()
            env.close()
        except Exception as e:
            logging.warning(f"L2 cache writer final flush failed: {e}")

def start_l2_writer(path: str, map_size_bytes: int) -> Tuple[mp.Queue, mp.Process]:
    """
    Starts and returns a dedicated L2 cache writer process and its command queue.

    Args:
        path: The directory path for the LMDB environment.
        map_size_bytes: The initial size of the LMDB memory map.

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
    proc = _WriterProc(path=path, map_size_bytes=map_size_bytes, q=q)
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
