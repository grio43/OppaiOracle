"""
Exclusion File Manager for OppaiOracle Training Pipeline

Handles reading, writing, and live-updating of the cache_exclusions.txt file
that tracks corrupted/failed images across training runs.

Features:
- Cross-platform file locking (Windows: msvcrt, Unix: fcntl)
- Deduplication on write
- Periodic reload for multi-worker synchronization
- Thread-safe operations

Usage:
    from utils.exclusion_manager import ExclusionManager

    # Create manager (typically one per dataset instance)
    mgr = ExclusionManager(data_root / 'cache_exclusions.txt')

    # Load existing exclusions
    excluded_ids = mgr.load()

    # Add a failed sample (immediately persisted)
    mgr.add_exclusion('image_12345')

    # Periodic reload to pick up exclusions from other workers
    mgr.reload_if_stale(interval_seconds=60)
"""

from __future__ import annotations
import os
import time
import logging
import platform
import threading
from pathlib import Path
from typing import Set, List, Optional

# Platform-specific file locking
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)


class ExclusionManager:
    """
    Manages the exclusion file for tracking failed/corrupted images.

    Supports:
    - Thread-safe loading and writing
    - Cross-platform file locking
    - Periodic reloading to sync across workers
    - Deduplication on write
    """

    def __init__(
        self,
        exclusion_path: Path,
        max_lock_retries: int = 3,
        reload_interval_seconds: float = 30.0,
    ):
        """
        Initialize the exclusion manager.

        Args:
            exclusion_path: Path to the exclusion file (e.g., data_root/cache_exclusions.txt)
            max_lock_retries: Maximum retries for acquiring file lock
            reload_interval_seconds: Minimum interval between reloads (default 30s)
        """
        self.exclusion_path = Path(exclusion_path)
        self.max_lock_retries = max_lock_retries
        self.reload_interval = reload_interval_seconds

        # In-memory cache
        self._excluded_ids: Set[str] = set()
        self._last_load_time: float = 0.0
        self._last_mtime: float = 0.0

        # Thread safety
        self._lock = threading.Lock()

        # Track pending writes for batching
        self._pending_writes: Set[str] = set()
        self._write_batch_size = 10  # Flush after this many pending writes

    def load(self) -> Set[str]:
        """
        Load exclusions from file.

        Returns:
            Set of excluded image IDs
        """
        with self._lock:
            return self._load_internal()

    def _load_internal(self) -> Set[str]:
        """Internal load (assumes lock is held)."""
        if not self.exclusion_path.exists():
            self._excluded_ids = set()
            self._last_load_time = time.time()
            return self._excluded_ids.copy()

        try:
            with open(self.exclusion_path, 'r', encoding='utf-8') as f:
                # Acquire shared lock for reading
                self._acquire_lock(f, exclusive=False)
                try:
                    self._excluded_ids = set()
                    for line in f:
                        line = line.strip()
                        if line:
                            # Handle both old format (full path) and new format (just id)
                            self._excluded_ids.add(Path(line).stem)
                    self._last_load_time = time.time()
                    try:
                        self._last_mtime = self.exclusion_path.stat().st_mtime
                    except OSError:
                        pass
                finally:
                    self._release_lock(f)

            if self._excluded_ids:
                logger.debug(f"Loaded {len(self._excluded_ids)} exclusions from {self.exclusion_path}")

        except Exception as e:
            logger.warning(f"Failed to load exclusions from {self.exclusion_path}: {e}")

        return self._excluded_ids.copy()

    def is_excluded(self, image_id: str) -> bool:
        """
        Check if an image ID is excluded.

        Args:
            image_id: Image ID (filename stem) to check

        Returns:
            True if excluded, False otherwise
        """
        return image_id in self._excluded_ids

    def add_exclusion(self, image_id: str, immediate: bool = False) -> bool:
        """
        Add an image ID to the exclusion list.

        Args:
            image_id: Image ID (filename stem) to exclude
            immediate: If True, write to disk immediately. If False, batch writes (default).
                      Batching avoids blocking fsync() calls during training.

        Returns:
            True if this was a new exclusion, False if already excluded
        """
        with self._lock:
            # Check if already excluded (in memory)
            if image_id in self._excluded_ids:
                return False

            # Add to in-memory set
            self._excluded_ids.add(image_id)

            if immediate:
                # Write immediately
                return self._write_exclusions_internal([image_id]) > 0
            else:
                # Batch for later
                self._pending_writes.add(image_id)
                if len(self._pending_writes) >= self._write_batch_size:
                    self._flush_pending_internal()
                return True

    def add_exclusions(self, image_ids: List[str]) -> int:
        """
        Add multiple image IDs to the exclusion list.

        Args:
            image_ids: List of image IDs to exclude

        Returns:
            Count of new exclusions added
        """
        with self._lock:
            # Filter to only new exclusions
            new_ids = [id for id in image_ids if id not in self._excluded_ids]
            if not new_ids:
                return 0

            # Add to in-memory set
            self._excluded_ids.update(new_ids)

            # Write to disk
            return self._write_exclusions_internal(new_ids)

    def flush_pending(self) -> int:
        """
        Flush any pending batched writes to disk.

        Returns:
            Count of exclusions written
        """
        with self._lock:
            return self._flush_pending_internal()

    def _flush_pending_internal(self) -> int:
        """Internal flush (assumes lock is held)."""
        if not self._pending_writes:
            return 0

        pending = list(self._pending_writes)
        self._pending_writes.clear()
        return self._write_exclusions_internal(pending)

    def _write_exclusions_internal(self, new_ids: List[str]) -> int:
        """
        Write exclusions to file with deduplication and locking.

        Args:
            new_ids: List of new image IDs to add

        Returns:
            Count of new entries actually written (excludes duplicates on disk)
        """
        if not new_ids:
            return 0

        try:
            # Create parent directory if needed
            self.exclusion_path.parent.mkdir(parents=True, exist_ok=True)

            # Open for read+write, create if needed
            mode = 'r+' if self.exclusion_path.exists() else 'w+'
            with open(self.exclusion_path, mode, encoding='utf-8') as f:
                # Acquire exclusive lock
                self._acquire_lock(f, exclusive=True)
                try:
                    # Read existing entries (in case other workers wrote while we waited)
                    f.seek(0)
                    existing = {line.strip() for line in f if line.strip()}

                    # Find truly new unique entries
                    new_unique = [id for id in new_ids if id not in existing]

                    if new_unique:
                        # Seek to end and append
                        f.seek(0, 2)
                        for image_id in new_unique:
                            f.write(image_id + '\n')
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk

                        logger.info(
                            f"Added {len(new_unique)} new exclusion(s) to {self.exclusion_path.name}"
                        )

                    return len(new_unique)
                finally:
                    self._release_lock(f)

        except Exception as e:
            logger.error(f"Failed to write exclusions to {self.exclusion_path}: {e}")
            return 0

    def reload_if_stale(self, force: bool = False) -> bool:
        """
        Reload exclusions from file if stale (file modified or interval elapsed).

        Args:
            force: Force reload regardless of staleness

        Returns:
            True if reloaded, False if still fresh
        """
        # Fast path: check time WITHOUT lock to avoid contention on every __getitem__
        # This is safe because _last_load_time is only updated while holding the lock
        if not force and (time.time() - self._last_load_time) < self.reload_interval:
            return False

        with self._lock:
            now = time.time()

            # Double-check after acquiring lock (another thread may have reloaded)
            if not force and (now - self._last_load_time) < self.reload_interval:
                return False

            # Check if file was modified
            if not force and self.exclusion_path.exists():
                try:
                    current_mtime = self.exclusion_path.stat().st_mtime
                    if current_mtime == self._last_mtime:
                        self._last_load_time = now  # Reset timer
                        return False
                except OSError:
                    pass

            # Reload
            old_count = len(self._excluded_ids)
            self._load_internal()
            new_count = len(self._excluded_ids)

            if new_count > old_count:
                logger.debug(
                    f"Reloaded exclusions: {new_count - old_count} new entries "
                    f"(total: {new_count})"
                )

            return True

    def get_excluded_ids(self) -> Set[str]:
        """
        Get a copy of the current excluded IDs set.

        Returns:
            Copy of excluded image IDs
        """
        with self._lock:
            return self._excluded_ids.copy()

    def _acquire_lock(self, f, exclusive: bool = True) -> bool:
        """
        Acquire file lock (cross-platform).

        Args:
            f: Open file handle
            exclusive: True for exclusive (write) lock, False for shared (read) lock

        Returns:
            True if lock acquired
        """
        if platform.system() == 'Windows':
            return self._acquire_lock_windows(f, exclusive)
        else:
            return self._acquire_lock_unix(f, exclusive)

    def _acquire_lock_windows(self, f, exclusive: bool) -> bool:
        """Windows-specific file locking with retry."""
        lock_size = max(1, f.seek(0, 2))  # Lock entire file
        f.seek(0)

        for retry in range(self.max_lock_retries):
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, lock_size)
                return True
            except (OSError, IOError) as e:
                if retry < self.max_lock_retries - 1:
                    time.sleep(0.1 * (2 ** retry))  # Exponential backoff
                else:
                    logger.debug(
                        f"Could not acquire file lock after {self.max_lock_retries} attempts: {e}"
                    )
                    return False
        return False

    def _acquire_lock_unix(self, f, exclusive: bool) -> bool:
        """Unix-specific file locking."""
        try:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(f, lock_type)
            return True
        except (OSError, IOError) as e:
            logger.debug(f"Failed to acquire Unix file lock: {e}")
            return False

    def _release_lock(self, f) -> None:
        """Release file lock (cross-platform)."""
        try:
            if platform.system() == 'Windows':
                lock_size = max(1, f.seek(0, 2))
                f.seek(0)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, lock_size)
                except (OSError, IOError):
                    pass  # Lock may already be released
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass  # Ignore unlock errors


# Standalone function for backward compatibility with l2_cache_warmup.py
def write_exclusions_deduplicated(
    exclusion_path: Path,
    new_ids: List[str],
    max_retries: int = 3
) -> int:
    """
    Write exclusions with deduplication and cross-platform file locking.

    This is a standalone function for backward compatibility.
    For new code, prefer using ExclusionManager class.

    Args:
        exclusion_path: Path to the exclusion file
        new_ids: List of new image IDs to add
        max_retries: Maximum number of retry attempts for locking

    Returns:
        Count of new entries actually added (excludes duplicates)
    """
    mgr = ExclusionManager(exclusion_path, max_lock_retries=max_retries)
    return mgr.add_exclusions(new_ids)
