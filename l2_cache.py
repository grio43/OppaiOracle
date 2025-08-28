from __future__ import annotations
import os, time, atexit, signal, queue
import multiprocessing as mp
from typing import Optional, Tuple
import lmdb
import torch
from cache_codec import encode_tensor as _tensor_to_bytes, decode_tensor as _tensor_from_bytes

# Serialization now lives in cache_codec.py (safetensors + optional HMAC)

# --- Read-only per-process cache handle ---
class LMDBReader:
    def __init__(self, path: str, map_size_bytes: int, max_readers: int = 4096):
        os.makedirs(path, exist_ok=True)
        # readonly=True avoids taking the write lock in workers
        self.env = lmdb.open(
            path,
            map_size=map_size_bytes,
            subdir=True,
            readonly=True,
            lock=True,          # keep OS locks; safe on local fs
            readahead=False,    # good for random access on SSD
            max_readers=max_readers,
            max_dbs=1
        )

    def get(self, key: bytes) -> Optional[bytes]:
        # short-lived read txns so the writer never waits on us
        with self.env.begin(write=False) as txn:
            return txn.get(key)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

# --- Dedicated writer process (single writer, batched commits) ---
class _WriterProc(mp.Process):
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

    def run(self):
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

        def _shutdown(*_):
            self._running.clear()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        pending = []
        pending_bytes = 0
        last_flush = time.time()

        while self._running.is_set():
            timeout = max(0.0, self.flush_s - (time.time() - last_flush))
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
                     or (time.time() - last_flush) >= self.flush_s)
            ):
                # Single short write txn → minimal time holding the writer lock
                with env.begin(write=True) as txn:
                    for k, v in pending:
                        txn.put(k, v, overwrite=True)
                pending.clear()
                pending_bytes = 0
                last_flush = time.time()

        # Final flush
        if pending:
            with env.begin(write=True) as txn:
                for k, v in pending:
                    txn.put(k, v, overwrite=True)
        try:
            env.sync(False)
            env.close()
        except Exception:
            pass

def start_l2_writer(path: str, map_size_bytes: int) -> Tuple[mp.Queue, mp.Process]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=16384)
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
