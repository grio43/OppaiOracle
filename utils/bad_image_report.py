"""Low-overhead, report-only tracking of failed images for one training process."""
from __future__ import annotations

import atexit
import logging
import os
import threading
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _field(value):
    return str(value).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')


class BadImageReporter:
    """One text row per path, across workers, epochs and normal restarts.

    The trainer submits only failed samples already returned by its DataLoader.
    No image reads, hashes, worker file locks, or synchronous disk writes occur
    in record_batch. The daemon batches appends every 30 seconds; only bad paths
    occupy the in-memory dedup set. The report never acts as an exclusion list.
    """

    def __init__(self, path: Path, flush_interval: float = 30.):
        self.path = Path(path)
        self.flush_interval = flush_interval
        self._lock = threading.Lock()
        self._pending = {}
        self._seen = set()
        self._stop = threading.Event()
        self._warned = False
        self._thread = threading.Thread(target=self._run, name='bad-image-report', daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def record_batch(self, batch):
        flags = batch.get('error')
        if flags is None:
            return
        paths = batch.get('error_path', batch.get('image_id', []))
        ids = batch.get('image_id', paths)
        reasons = batch.get('error_reason', [''] * len(paths))
        with self._lock:
            for i in flags.nonzero(as_tuple=True)[0].tolist():
                path = _field(paths[i])
                key = os.path.normcase(os.path.normpath(path))
                if key not in self._seen:
                    self._seen.add(key)
                    self._pending[key] = f'{path}\t{_field(ids[i])}\t{_field(reasons[i])}\n'

    def _run(self):
        # Windows training workstation: lower only this writer thread's priority.
        if os.name == 'nt':
            try:
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.GetCurrentThread.restype = wintypes.HANDLE
                kernel.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
                kernel.SetThreadPriority(kernel.GetCurrentThread(), -2)  # THREAD_PRIORITY_LOWEST
            except Exception:
                pass
        written = set()
        try:
            if self.path.exists():
                with self.path.open(encoding='utf-8') as stream:
                    for line in stream:
                        path = line.split('\t', 1)[0].rstrip('\r\n')
                        if path:
                            written.add(os.path.normcase(os.path.normpath(path)))
            with self._lock:
                self._seen.update(written)
            while not self._stop.wait(self.flush_interval):
                self._flush(written)
            self._flush(written)
        except Exception as exc:
            LOGGER.warning('Bad-image report unavailable (%s); training continues: %s', self.path, exc)

    def _flush(self, written):
        with self._lock:
            pending, self._pending = self._pending, {}
        new_rows = {key: row for key, row in pending.items() if key not in written}
        if not new_rows:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.writelines(new_rows.values())
            written.update(new_rows)
        except OSError as exc:
            # Reporting is advisory. Keep pending records for a later retry;
            # never stall or abort the optimizer for a log-file error.
            with self._lock:
                self._pending.update(new_rows)
            if not self._warned:
                LOGGER.warning('Could not append bad-image report; training continues: %s', exc)
                self._warned = True

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.)
        atexit.unregister(self.close)
