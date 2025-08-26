#!/usr/bin/env python3
"""
Data loading and augmentation utilities for the anime tagger.

This module provides a simplified HDF5/JSON loader with support for
letterbox-style resizing.  Images are resized to fit within a square
canvas while preserving aspect ratio and padded with a neutral colour.
Padding information is returned so downstream modules can mask out
padded regions (e.g. during vision transformer patchification).

The default pad colour is a mid‑grey (114,114,114) as commonly used by
YOLO models.  The pad colour and patch size are configurable via
``SimplifiedDataConfig``.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy
import multiprocessing as mp
import queue
import sqlite3
import lmdb
from pathlib import Path
import hashlib
import json
import warnings
import yaml
import threading
import logging
from datetime import datetime, timedelta
import random
import psutil
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler, Sampler
import torchvision.transforms as T
import torchvision.transforms.v2 as v2
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import torchvision.transforms.functional as TF
from PIL import Image

from collections import OrderedDict

# Optional compression support
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


from orientation_handler import OrientationHandler, OrientationMonitor  # type: ignore
from Configuration_System import DataConfig, ValidationConfig
# Import TagVocabulary from the vocabulary module rather than a relative package path.
# The vocabulary module should reside on the Python path for this import to succeed.
from vocabulary import TagVocabulary

logger = logging.getLogger(__name__)

# ---- RNG helpers ------------------------------------------------------------
def _derive_worker_generator(worker_id: int = 0) -> torch.Generator:
    """Derive a worker-specific generator from torch.initial_seed() and worker_id."""
    return torch.Generator().manual_seed(torch.initial_seed() + worker_id)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
# The default vocabulary path is now determined by the calling script
# from the unified configuration and passed into create_dataloaders.

@dataclass
class AugmentationStats:
    """Statistics for data augmentation operations."""
    # Flip stats
    flip_total: int = 0
    flip_safe: int = 0
    flip_skipped_text: int = 0
    flip_skipped_unmapped: int = 0
    flip_blocked_safety: int = 0

    # Color jitter stats
    jitter_applied: int = 0
    jitter_brightness_factors: List[float] = field(default_factory=list)
    jitter_contrast_factors: List[float] = field(default_factory=list)
    jitter_saturation_factors: List[float] = field(default_factory=list)
    jitter_hue_factors: List[float] = field(default_factory=list)

    # Crop stats
    crop_applied: int = 0
    crop_scales: List[float] = field(default_factory=list)
    crop_aspects: List[float] = field(default_factory=list)

    # Resize stats
    resize_scales: List[float] = field(default_factory=list)
    resize_pad_pixels: List[int] = field(default_factory=list)

    # Batch info
    batch_count: int = 0
    image_count: int = 0

    def aggregate(self, other: 'AugmentationStats'):
        """Aggregate stats from another instance."""
        self.flip_total += other.flip_total
        self.flip_safe += other.flip_safe
        self.flip_skipped_text += other.flip_skipped_text
        self.flip_skipped_unmapped += other.flip_skipped_unmapped
        self.flip_blocked_safety += other.flip_blocked_safety
        self.jitter_applied += other.jitter_applied
        self.crop_applied += other.crop_applied
        self.batch_count += other.batch_count
        self.image_count += other.image_count
        # Note: Don't aggregate the lists as they would grow unbounded

class BoundedLevelAwareQueue:
    """Bounded queue with level-aware dropping policy for logging."""
    
    def __init__(self, maxsize=5000):
        self.maxsize = maxsize
        self.queue = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.drop_counts = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
        
    def put(self, record, block=False, timeout=None):
        """Put record with level-aware dropping."""
        with self.lock:
            if len(self.queue) >= self.maxsize:
                # Drop policy: prefer dropping DEBUG/INFO over WARNING/ERROR/CRITICAL
                if hasattr(record, 'levelname'):
                    if record.levelname in ('DEBUG', 'INFO'):
                        # Drop this low-priority message
                        self.drop_counts[record.levelname] += 1
                        return
                    else:
                        # Try to drop a DEBUG/INFO message to make room
                        for i, item in enumerate(self.queue):
                            if hasattr(item, 'levelname') and item.levelname in ('DEBUG', 'INFO'):
                                self.queue.pop(i)
                                self.drop_counts[item.levelname] += 1
                                break
                        else:
                            # No DEBUG/INFO to drop, drop oldest
                            if self.queue:
                                dropped = self.queue.pop(0)
                                if hasattr(dropped, 'levelname'):
                                    self.drop_counts[dropped.levelname] += 1
            
            self.queue.append(record)
            self.not_empty.notify()
    
    def get(self, block=True, timeout=None):
        """Get record from queue."""
        with self.lock:
            if not block:
                if not self.queue:
                    raise queue.Empty
                return self.queue.pop(0)
            
            deadline = None if timeout is None else time.time() + timeout
            while not self.queue:
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self.not_empty.wait(remaining)
            return self.queue.pop(0)

    def put_nowait(self, record):
        """Non-blocking put for QueueHandler compatibility."""
        return self.put(record, block=False)
    
    def get_nowait(self):
        """Non-blocking get for consistency."""
        return self.get(block=False)
    
    def get_drop_stats(self):
        """Get drop statistics."""
        with self.lock:
            return self.drop_counts.copy()


class CompressingRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler with optional compression."""
    
    def __init__(self, *args, compress=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.compress = compress and LZ4_AVAILABLE
        
    def doRollover(self):
        """Override to add compression after rotation."""
        super().doRollover()
        
        if self.compress and self.backupCount > 0:
            # Compress the most recent backup
            for i in range(self.backupCount, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                if os.path.exists(sfn) and not sfn.endswith('.lz4'):
                    self._compress_file(sfn)
    
    def _compress_file(self, filepath):
        """Compress a file using lz4."""
        try:
            import lz4.frame
            compressed_path = filepath + '.lz4'
            with open(filepath, 'rb') as f_in:
                with lz4.frame.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(filepath)
            logger.debug(f"Compressed {filepath} -> {compressed_path}")
        except Exception as e:
            logger.warning(f"Failed to compress {filepath}: {e}")


def setup_worker_logging(worker_id: int, log_queue: object):
    """Setup worker logging via queue only - no per-worker files."""
    # Get worker-specific logger to avoid polluting root logger
    worker_logger_name = f'oppai.worker.{worker_id}'
    worker_logger = logging.getLogger(worker_logger_name)
    worker_logger.propagate = False  # Don't propagate to root
    worker_logger.setLevel(logging.INFO)

    # Check if handlers already exist (idempotency)
    existing_queue_handlers = [h for h in worker_logger.handlers if isinstance(h, QueueHandler)]
    if existing_queue_handlers:
        # Already configured, skip
        return

    # Only add queue handler - main process handles file writing
    queue_handler = QueueHandler(log_queue)
    queue_handler.setLevel(logging.INFO)
    worker_logger.addHandler(queue_handler)

    # Also configure the module logger to use the queue
    module_logger = logging.getLogger(__name__)
    # Remove any existing handlers to avoid duplicates
    module_logger.handlers = [h for h in module_logger.handlers 
                              if not isinstance(h, (QueueHandler, CompressingRotatingFileHandler))]
    module_logger.addHandler(queue_handler)
    module_logger.setLevel(logging.INFO)

    logger.debug(f"Worker {worker_id} logging configured (queue-only)")

def letterbox_resize(
    image: torch.Tensor,
    target_size: int,
    pad_color: Iterable[int] = (114, 114, 114),
    patch_size: int = 16,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Resize an image tensor to a square canvas while preserving aspect ratio.

    The input image is first scaled by the minimal factor required to fit
    within the ``target_size``.  Any remaining space is padded equally on
    each side using the provided ``pad_color``.  If the resulting
    dimensions are not divisible by ``patch_size`` then additional padding
    is applied to the bottom and right edges so that the final width and
    height are multiples of ``patch_size``.  This ensures that when the
    image is partitioned into non‑overlapping patches (e.g. for a vision
    transformer) no partial patches are dropped.

    Args:
        image: Input image tensor of shape (C, H, W) with values in [0, 1].
        target_size: Desired square size (both height and width) of the output.
        pad_color: RGB colour used to fill padded regions.  Values should be
            integers in the 0‑255 range.
        patch_size: Patch size used by the downstream model.  The output
            dimensions will be rounded up to the next multiple of this value.

    Returns:
        A tuple containing:
          * The padded image tensor of shape (C, H_out, W_out).
          * A dictionary with keys ``scale`` and ``pad`` describing the
            applied scaling factor and padding on each side (left, top,
            right, bottom).  These values can be used to derive padding
            masks during patchification.
    """
    c, h, w = image.shape
    # Compute scaling factor to fit the longer side into target_size.
    r = min(target_size / float(h), target_size / float(w))
    # Avoid degenerate new sizes.
    new_h = int(round(h * r))
    new_w = int(round(w * r))
    # Resize using bilinear interpolation on tensors.
    resized = TF.resize(image, [new_h, new_w], interpolation=T.InterpolationMode.BILINEAR)
    # Compute padding needed to reach target_size.
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # After initial letterbox, ensure divisibility by patch_size.
    final_h = target_size
    final_w = target_size
    extra_pad_bottom = 0
    extra_pad_right = 0
    # Only apply divisibility padding when patch_size > 1; None is treated as no divisibility requirement
    if patch_size > 1:
        if final_h % patch_size != 0:
            extra_pad_bottom = patch_size - (final_h % patch_size)
        if final_w % patch_size != 0:
            extra_pad_right = patch_size - (final_w % patch_size)
    final_h += extra_pad_bottom
    final_w += extra_pad_right
    # Create canvas and fill with pad colour.
    canvas = torch.zeros((c, final_h, final_w), dtype=resized.dtype)
    # Normalise pad colour to [0,1] range.
    pad_vals = torch.tensor(pad_color, dtype=resized.dtype) / 255.0
    for ch in range(c):
        canvas[ch, :, :] = pad_vals[ch]
    # Paste resized image into the centre region.
    start_y = pad_top
    end_y = pad_top + new_h
    start_x = pad_left
    end_x = pad_left + new_w
    canvas[:, start_y:end_y, start_x:end_x] = resized
    info = {
        "scale": r,
        "pad": (pad_left, pad_top, pad_right + extra_pad_right, pad_bottom + extra_pad_bottom),
        "out_size": (final_h, final_w),
        "in_size": (h, w),
    }
    return canvas, info


class TrackedColorJitter:
    """Wrapper for ColorJitter that tracks sampled parameters."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,
                 generator: Optional[torch.Generator] = None,
                 max_hue_for_eyes: float = 0.03):
        # Torchvision op used for application; params sampled by us (from torch RNG)
        self._jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.brightness_range = float(brightness or 0.0)
        self.contrast_range   = float(contrast or 0.0)
        self.saturation_range = float(saturation or 0.0)
        # Cap hue at configurable threshold for semantic safety (eye-color)
        self.hue_range        = float(min(hue or 0.0, max_hue_for_eyes))
        self.last_params = {}
        self._external_gen = generator
        self._worker_id = None  # Will be set if in worker process

    def _gen(self) -> torch.Generator:
        if self._external_gen is not None:
            return self._external_gen
        # Use worker-specific generator if available
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return _derive_worker_generator(worker_info.id)
        return _derive_worker_generator(0)

    def __call__(self, img):
        """Apply jitter deterministically via torch.Generator and track params."""
        g = self._gen()
        # brightness ∈ [1-b, 1+b], contrast ∈ [1-c, 1+c], saturation ∈ [1-s, 1+s]
        def _span(center: float, width: float) -> float:
            if width <= 0: return center
            lo = max(0.0, center - width)
            hi = center + width
            return (hi - lo) * torch.rand((), generator=g).item() + lo

        brightness_factor = _span(1.0, self.brightness_range)
        contrast_factor   = _span(1.0, self.contrast_range)
        saturation_factor = _span(1.0, self.saturation_range)
        hue_span          = self.hue_range
        hue_factor        = 0.0 if hue_span <= 0 else (2.0 * torch.rand((), generator=g).item() - 1.0) * hue_span

        # Record for stats/telemetry
        self.last_params = {
            'brightness': brightness_factor,
            'contrast':   contrast_factor,
            'saturation': saturation_factor,
            'hue':        hue_factor,
        }

        # Apply using torchvision.functional in a fixed order matching ColorJitter
        out = TF.adjust_brightness(img, brightness_factor)
        out = TF.adjust_contrast(out, contrast_factor)
        out = TF.adjust_saturation(out, saturation_factor)
        if hue_span > 0:
            out = TF.adjust_hue(out, hue_factor)
        return out

# The SimplifiedDataConfig dataclass has been removed.
# The create_dataloaders function now accepts DataConfig and ValidationConfig objects
# from Configuration_System.py.


def _make_worker_init_fn(base_seed: Optional[int], log_queue):
    """
    Seed torch/numpy/random per worker and disable file logging in workers.
    If a logging queue is provided, attach a QueueHandler so logs go to main.
    """
    def init_fn(worker_id: int):
        # Derive from torch.initial_seed() so each worker & epoch stream is unique
        torch_seed = torch.initial_seed() + worker_id
        random.seed(torch_seed)
        np.random.seed(torch_seed % (2**32 - 1))
        torch.manual_seed(torch_seed)

        # Setup worker logging with rotation
        if log_queue is not None:
            setup_worker_logging(worker_id, log_queue)

        # Important: Close any inherited LMDB environments in worker (run once)
        # This prevents issues with shared file descriptors
        # Use a flag to ensure this only runs once per process
        if not hasattr(init_fn, '_lmdb_cleaned'):
            init_fn._lmdb_cleaned = set()

        current_pid = os.getpid()
        if current_pid not in init_fn._lmdb_cleaned:
            init_fn._lmdb_cleaned.add(current_pid)
            
            # Suppress deprecation warnings during the GC scan
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                import gc
                for obj in gc.get_objects():
                    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'LMDBCache':
                        if hasattr(obj, 'env') and obj.env is not None:
                            try:
                                obj.env.close()
                            except Exception:
                                pass

        logger.debug(f"Worker {worker_id} seeded")

    return init_fn


# DataLoader construction lives below...
#  - Seeds Python/NumPy per worker from torch.initial_seed()
#  - Wires optional generator so transforms see independent RNG streams

class LMDBCache:
    """Global file-backed LMDB cache (L2) with memory-mapped access"""
    _logged_pids = set()

    def __init__(
        self,
        path: Path,
        max_size_gb: float = 48.0,
        map_growth_gb: float = 4.0,
        readonly: bool = False,
        max_readers: int = 2048,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.map_growth_bytes = int(map_growth_gb * 1024 ** 3)
        # Pre-allocate full size to avoid MDB_MAP_RESIZED errors
        self.current_map_size = self.max_size_bytes
        self.readonly = readonly
        self.max_readers = max_readers
        self.resize_attempted = False  # Track if we've already tried resizing

        # Open LMDB environment lazily and track process
        self._env = None
        self._pid = None
        self._open_env()

        # Track cache statistics
        self.hits = 0
        self.misses = 0
        self.duplicate_keys = 0
        self.rejection_count = 0
        self.size_bytes = 0
        self._lock = threading.Lock()
        self.grace_margin = 0.05  # 5% grace margin
        self.max_value_bytes = int(32 * 1024 * 1024)  # 32MB max per value

    def _open_env(self):
        """Open or reopen LMDB environment for current process with full size pre-allocated"""
        # Check actual database size if it exists
        data_file = self.path / "data.mdb"
        if data_file.exists():
            actual_size = data_file.stat().st_size
            if actual_size > self.current_map_size:
                logger.warning(
                    f"LMDB database size ({actual_size / (1024**3):.1f}GB) exceeds configured map size "
                    f"({self.current_map_size / (1024**3):.1f}GB), adjusting"
                )
                self.current_map_size = int(actual_size * 1.5)

        self.env = lmdb.open(
            str(self.path),
            map_size=self.current_map_size,  # Now pre-allocated to max
            readonly=self.readonly,
            lock=not self.readonly,
            max_dbs=1,
            writemap=not self.readonly,  # Only use writemap for writers
            readahead=False,  # Disable readahead for better memory control
            metasync=False,  # Don't sync metadata for each transaction
            sync=False,  # Don't sync data for each transaction (rely on OS)
            map_async=True,  # Allow async writes
            max_readers=self.max_readers,
        )
        self._pid = os.getpid()
        if self._pid not in LMDBCache._logged_pids:
            logger.info(f"LMDB environment opened with map_size={self.current_map_size / (1024**3):.1f}GB for process {self._pid}")
            LMDBCache._logged_pids.add(self._pid)
        else:
            logger.debug(f"LMDB environment re-opened for process {self._pid}")

    def _ensure_env(self):
        """Ensure LMDB environment is open for current process"""
        current_pid = os.getpid()
        if self._pid != current_pid:
            if hasattr(self, 'env') and self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            self._open_env()

    def _handle_map_resized(self):
        """Handle MDB_MAP_RESIZED error by reopening with larger size"""
        if self.resize_attempted:
            # Already tried once, don't retry indefinitely
            return False

        self.resize_attempted = True
        logger.warning("MDB_MAP_RESIZED encountered, reopening environment with larger map size")

        # Close current environment
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        # Reopen with increased size
        self.current_map_size = self.max_size_bytes
        self._open_env()
        return True

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get item from cache"""
        self._ensure_env()
        txn = None
        try:
            txn = self.env.begin(buffers=True)
            data = txn.get(key.encode())
            txn.abort()  # explicitly close read transaction
            txn = None
        except lmdb.MapResizedError:
            if txn is not None:
                txn.abort()
            if self._handle_map_resized():
                # Retry once after resize
                return self.get(key)
            return None
        except Exception as e:
            logger.debug(f"LMDB get error for key {key}: {e}")
            return None
        finally:
            if txn is not None:
                txn.abort()

        try:
            if data is not None:
                self.hits += 1
                # Deserialize tensor.  ``np.frombuffer`` returns a read-only array
                # which subsequently triggers a PyTorch warning when wrapping it
                # with ``torch.from_numpy``. Copy the buffer to ensure writability
                # and avoid the warning about undefined behaviour when tensors are
                # created from non-writable NumPy arrays.
                buffer = np.frombuffer(data, dtype=np.uint8).copy()
                tensor = torch.from_numpy(buffer)
                return tensor
            else:
                self.misses += 1
                return None
        except Exception as e:
            logger.error(f"Error deserializing cached tensor: {e}")
            return None
    
    def put(self, key: str, value: torch.Tensor, check_memory: bool = True) -> bool:
        """Put item in cache with memory checking"""
        with self._lock:
            if self.readonly:
                return False

            self._ensure_env()

            # Check if key already exists (track duplicates)
            txn = None
            try:
                txn = self.env.begin(buffers=True)
                exists = txn.get(key.encode()) is not None
                txn.abort()
            finally:
                if txn is not None:
                    txn.abort()
            if exists:
                self.duplicate_keys += 1
                return True  # Already cached

            # Serialize tensor with canonical dtype
            if value.dtype != torch.uint8:
                value = (value * 255).to(torch.uint8) if value.dtype.is_floating_point else value.to(torch.uint8)

            value_np = value.cpu().numpy()
            value_bytes = value_np.tobytes()

            # Check value size limit
            if len(value_bytes) > self.max_value_bytes:
                logger.warning(
                    f"Value too large ({len(value_bytes)/1024/1024:.1f}MB > {self.max_value_bytes/1024/1024:.1f}MB), rejecting"
                )
                self.rejection_count += 1
                return False

            # Check capacity with grace margin
            txn = None
            try:
                txn = self.env.begin()
                stat = txn.stat()
                txn.abort()
            finally:
                if txn is not None:
                    txn.abort()
            current_size = stat['psize'] * stat['leaf_pages']

            max_allowed = self.max_size_bytes * (1 - self.grace_margin)
            if current_size + len(value_bytes) > max_allowed:
                logger.warning(
                    f"Cache approaching capacity limit with grace margin, rejecting insert"
                )
                self.rejection_count += 1
                return False

            # Check memory watermarks if requested
            if check_memory and not self._check_memory_available():
                return False

        try:
            txn = None
            try:
                txn = self.env.begin(write=True)
                txn.put(key.encode(), value_bytes)
                txn.commit()
                txn = None
            finally:
                if txn is not None:
                    txn.abort()
            return True
        except lmdb.MapResizedError:
            if txn is not None:
                txn.abort()
            if self._handle_map_resized():
                # Retry once after resize
                return self.put(key, value, check_memory)
            logger.warning("LMDB put failed due to map resize")
            return False
        except lmdb.MapFullError:
            # Try to grow the map
            if self._grow_map():
                # Retry after growing
                try:
                    txn = None
                    try:
                        txn = self.env.begin(write=True)
                        txn.put(key.encode(), value_bytes)
                        txn.commit()
                        txn = None
                    finally:
                        if txn is not None:
                            txn.abort()
                    return True
                except lmdb.MapFullError:
                    logger.warning("LMDB cache is full even after growth")
                    return False
            else:
                logger.warning("LMDB cache is full and cannot grow further")
                return False
    
    def _grow_map(self) -> bool:
        """Try to grow the LMDB map size."""
        self._ensure_env()
        if self.current_map_size >= self.max_size_bytes:
            return False

        new_size = min(self.current_map_size + self.map_growth_bytes, self.max_size_bytes)
        try:
            self.env.set_mapsize(new_size)
            self.current_map_size = new_size
            logger.info(f"LMDB map grown to {new_size / (1024**3):.1f} GB")
            return True
        except Exception as e:
            logger.error(f"Failed to grow LMDB map: {e}")
            return False
    
    def _check_memory_available(self) -> bool:
        """Check if we have enough free memory"""
        mem = psutil.virtual_memory()
        free_pct = (mem.available / mem.total) * 100
        
        # Three-tier watermark system
        if free_pct < 5.0:  # Critical
            logger.warning(f"Critical memory pressure: {free_pct:.1f}% free")
            return False
        elif free_pct < 12.0:  # Low
            # Throttle writes
            import time
            time.sleep(0.01)  # Small delay to reduce pressure
            return True
        elif free_pct < 25.0:  # High
            # Continue but monitor closely
            return True
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._ensure_env()
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / max(1, total)

            # Get LMDB stats
            txn = None
            try:
                txn = self.env.begin()
                stat = txn.stat()
                txn.abort()
            finally:
                if txn is not None:
                    txn.abort()

            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'duplicate_keys': self.duplicate_keys,
                'duplicate_rate': self.duplicate_keys / max(1, self.hits + self.misses),
                'entries': stat['entries'],
                'size_bytes': stat['psize'] * stat['leaf_pages'],
                'rejections': self.rejection_count,
            }
    
    def close(self):
        """Close the LMDB environment"""
        if hasattr(self, 'env'):
            self.env.close()


class ValidationIndex:
    """SQLite-based validation index for lazy validation with retry/backoff/TTL"""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = 3
        self.ttl_days = 7
        self.backoff_base = 60  # seconds        
        
        # Create/open database
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.lock = threading.Lock()
        
        # Create table if not exists
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_status (
                    image_path TEXT PRIMARY KEY,
                    status TEXT,
                    last_checked TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    next_retry_after TIMESTAMP,
                    first_failure TIMESTAMP
                )
            """)
            self.conn.commit()
    
    def get_status(self, image_path: str) -> Optional[str]:
        """Get validation status for an image"""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT status FROM validation_status WHERE image_path = ?",
                (image_path,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        
    def should_retry(self, image_path: str) -> bool:
        """Check if image should be retried based on backoff/TTL."""
        with self.lock:
            cursor = self.conn.execute(
                """SELECT retry_count, next_retry_after, first_failure 
                   FROM validation_status WHERE image_path = ?""",
                (image_path,)
            )
            result = cursor.fetchone()
            if not result:
                return True
            
            retry_count, next_retry_after, first_failure = result
            
            # Check TTL
            if first_failure:
                from datetime import datetime, timedelta
                first_fail_dt = datetime.fromisoformat(first_failure)
                if datetime.now() - first_fail_dt > timedelta(days=self.ttl_days):
                    # Expired, skip permanently
                    return False
            
            # Check backoff
            if next_retry_after:
                next_retry_dt = datetime.fromisoformat(next_retry_after)
                if datetime.now() < next_retry_dt:
                    return False
            
            return retry_count < self.max_retries
    
    def set_status(self, image_path: str, status: str, error_msg: Optional[str] = None):
        """Set validation status for an image"""
        from datetime import timedelta

        # Calculate next retry time with exponential backoff
        next_retry = None
        if status == 'failed':
            with self.lock:
                cursor = self.conn.execute(
                    "SELECT retry_count FROM validation_status WHERE image_path = ?",
                    (image_path,)
                )
                result = cursor.fetchone()
                retry_count = result[0] if result else 0
                backoff_seconds = self.backoff_base * (2 ** retry_count)
                next_retry = datetime.now() + timedelta(seconds=backoff_seconds)

        with self.lock:
            self.conn.execute(
                """INSERT INTO validation_status 
                   (image_path, status, last_checked, error_message, retry_count, next_retry_after, first_failure)
                   VALUES (?, ?, ?, ?, 1, ?, ?)
                   ON CONFLICT(image_path) DO UPDATE SET
                   status = excluded.status,
                   last_checked = excluded.last_checked,
                   error_message = excluded.error_message,
                   retry_count = retry_count + 1,
                   next_retry_after = ?,
                   first_failure = COALESCE(first_failure, excluded.first_failure)""",
                (image_path, status, datetime.now(), error_msg, next_retry,
                 datetime.now() if status == 'failed' else None, next_retry)
            )
            self.conn.commit()
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

class BackgroundValidator(threading.Thread):
    """Background thread for opportunistic validation."""
    
    def __init__(self, validation_index: ValidationIndex, data_dir: Path):
        super().__init__(daemon=True, name="BackgroundValidator")
        self.validation_index = validation_index
        self.data_dir = data_dir
        self.running = False
        self.stop_event = threading.Event()
        
    def run(self):
        """Run background validation loop."""
        self.running = True
        while self.running and not self.stop_event.is_set():
            # Get a batch of unvalidated or retry-eligible images
            # (Implementation would query the validation index for candidates)
            
            # Sleep between validation attempts
            self.stop_event.wait(timeout=30)
    
    def stop(self):
        """Stop the background validator."""
        self.running = False
        self.stop_event.set()


class AdaptivePrefetchController:
    """Controls prefetch factor and pinned memory based on memory pressure."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.original_prefetch = config.get('prefetch_factor', 2)
        self.max_inflight_per_worker = config.get('max_inflight_per_worker', 2)
        self.global_max_inflight = config.get('global_max_inflight', 16)
        self.cooldown_counter = 0
        self.emergency_unpin_threshold = 5.0  # Critical memory %
        self.reduce_prefetch_threshold = 12.0  # Low memory %
        self.restore_threshold = 25.0  # High memory %
        self.current_inflight = 0
        self.lock = threading.Lock()
        self.pinned_bytes = 0
        
    def check_and_adapt(self, loader: DataLoader) -> Dict[str, Any]:
        """Adapt prefetch and pinning based on memory pressure."""
        mem = psutil.virtual_memory()
        free_pct = (mem.available / mem.total) * 100
        
        actions = {'pin_memory': loader.pin_memory, 'prefetch_factor': loader.prefetch_factor}
        
        with self.lock:
            if free_pct < self.emergency_unpin_threshold:
                # Emergency unpin
                if loader.pin_memory:
                    logger.warning(f"Critical memory ({free_pct:.1f}%), disabling pinned memory")
                    loader.pin_memory = False
                    actions['pin_memory'] = False
                    self.cooldown_counter = 300
                    self.pinned_bytes = 0
                    
            elif free_pct < self.reduce_prefetch_threshold:
                # Reduce prefetch
                if loader.prefetch_factor > 1:
                    logger.info(f"Low memory ({free_pct:.1f}%), reducing prefetch")
                    loader.prefetch_factor = 1
                    actions['prefetch_factor'] = 1
                    
            elif free_pct > self.restore_threshold and self.cooldown_counter == 0:
                # Restore settings
                if loader.prefetch_factor < self.original_prefetch:
                    loader.prefetch_factor = min(
                        self.original_prefetch,
                        self._calculate_safe_prefetch()
                    )
                    actions['prefetch_factor'] = loader.prefetch_factor
                    
                if not loader.pin_memory and self.config.get('enabled', True):
                    loader.pin_memory = True
                    actions['pin_memory'] = True
            
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                
        return actions
    
    def _calculate_safe_prefetch(self) -> int:
        """Calculate safe prefetch factor based on in-flight limits."""
        with self.lock:
            # Respect per-worker and global limits
            return min(
                self.original_prefetch,
                self.max_inflight_per_worker,
                max(1, self.global_max_inflight // max(1, self.current_inflight))
            )
    
    def update_inflight(self, num_workers: int):
        """Update current in-flight worker count."""
        with self.lock:
            self.current_inflight = num_workers
    
    def track_pinned_bytes(self, batch_size: int, element_size: int):
        """Track pinned memory usage."""
        with self.lock:
            self.pinned_bytes = batch_size * element_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get prefetch controller metrics."""
        with self.lock:
            return {
                'pinned_bytes': self.pinned_bytes,
                'current_inflight': self.current_inflight,
                'cooldown_remaining': self.cooldown_counter
            }

class WorkingSetSampler:
    """Two-stage working set sampler with bounded trickle-in"""
    
    def __init__(
        self,
        dataset_size: int,
        working_set_pct: float = 5.0,
        max_items: int = 400000,
        trickle_pct: float = 1.0,
        max_new_per_epoch: int = 80000,
        refresh_epochs: int = 2,
        sample_weights: Optional[np.ndarray] = None
    ):
        self.dataset_size = dataset_size
        self.working_set_size = min(
            int(dataset_size * working_set_pct / 100),
            max_items
        )
        self.trickle_size = min(
            int(dataset_size * trickle_pct / 100),
            max_new_per_epoch
        )
        self.refresh_epochs = refresh_epochs
        self.sample_weights = sample_weights
        
        # Initialize working set
        self.working_set = set()
        self.unused_indices = set(range(dataset_size))
        self.epoch_count = 0
        
        # Initialize with random subset
        self._refresh_working_set()
        
        logger.info(
            f"WorkingSetSampler initialized: "
            f"working_set={self.working_set_size}, "
            f"trickle={self.trickle_size}"
        )
    
    def _refresh_working_set(self):
        """Refresh the working set"""
        # Sample new working set based on weights if provided
        if self.sample_weights is not None:
            # Weight-based sampling for working set
            probs = self.sample_weights / self.sample_weights.sum()
            indices = np.random.choice(
                self.dataset_size,
                size=self.working_set_size,
                replace=False,
                p=probs
            )
        else:
            # Random sampling
            indices = np.random.choice(
                self.dataset_size,
                size=self.working_set_size,
                replace=False
            )
        
        self.working_set = set(indices)
        self.unused_indices = set(range(self.dataset_size)) - self.working_set
    
    def get_epoch_indices(self) -> List[int]:
        """Get indices for current epoch with trickle-in"""
        # Check if we should refresh
        if self.epoch_count > 0 and self.epoch_count % self.refresh_epochs == 0:
            self._refresh_working_set()
        
        # Start with working set
        epoch_indices = list(self.working_set)
        
        # Add trickle-in of new samples
        if self.unused_indices and self.trickle_size > 0:
            new_samples = min(self.trickle_size, len(self.unused_indices))
            trickle_indices = random.sample(list(self.unused_indices), new_samples)
            epoch_indices.extend(trickle_indices)
            
            # Move trickled samples to working set for next epoch
            for idx in trickle_indices:
                self.unused_indices.remove(idx)
                # Optionally add to working set (with eviction if needed)
                if len(self.working_set) >= self.working_set_size:
                    # Evict random sample from working set
                    evict = random.choice(list(self.working_set))
                    self.working_set.remove(evict)
                    self.unused_indices.add(evict)
                self.working_set.add(idx)
        
        self.epoch_count += 1
        return epoch_indices


class EpochAwareWeightedSampler(Sampler):
    """Weighted sampler that properly reshuffles each epoch.

    This sampler ensures different sampling patterns each epoch while
    maintaining reproducibility through epoch-specific seeding.
    """

    def __init__(self, weights, num_samples, replacement=True, base_seed=None):
        """Initialize the epoch-aware weighted sampler.

        Args:
            weights: Sampling weights for each sample
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
            base_seed: Base seed for reproducibility
        """
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.base_seed = base_seed if base_seed is not None else int(torch.empty((), dtype=torch.int64).random_().item())
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self):
        # Create epoch-specific generator for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch)

        # Sample with epoch-specific randomness
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=generator
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

def _compute_cache_key(
    image_path: str,
    transform_signature: Optional[str] = None,
    target_shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[str] = None
) -> str:
    """
    Compute deduplicated cache key including transform parameters and file modification time.
    """
    try:
        # Include file modification time to invalidate cache if the file changes
        mtime = os.path.getmtime(image_path)
    except OSError:
        # Handle cases where the file might not exist or is inaccessible
        mtime = -1

    key_parts = [image_path, str(mtime)]
    
    if transform_signature:
        key_parts.append(f"transform_{transform_signature}")
    if target_shape:
        key_parts.append(f"shape_{target_shape}")
    if dtype:
        key_parts.append(f"dtype_{dtype}")
    
    # Create hash for compact key
    key_str = "|".join(str(p) for p in key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()



class SimplifiedDataset(Dataset):
    """Dataset for anime image tagging with augmentation and sampling.

    Each item returned is a dictionary containing an image tensor,
    multi‑hot tag labels, a rating label and some metadata (index, path,
    original tag list and rating string).  Letterbox resizing is applied
    on the fly to preserve the original aspect ratio and avoid dropping
    edge pixels when splitting into patches.  Padding information is
    returned in the metadata for use by downstream modules.
    """

    def __init__(
            self,
            config: DataConfig,
            json_files: List[Path],
            split: str,
            vocab: TagVocabulary,
        ) -> None:
            # Extract unpicklable objects before copying
            stats_queue = config.stats_queue if hasattr(config, 'stats_queue') else None

            # Create a shallow copy of config to avoid mutation by downstream components
            # Deep copy fails with multiprocessing.Queue objects
            if isinstance(config, DataConfig):
                # For dataclass, create a new instance with same values
                config = DataConfig(**{f.name: getattr(config, f.name)
                                                for f in type(config).__dataclass_fields__.values() if f.init})
            else:
                config = copy.copy(config)

            config.stats_queue = stats_queue  # Restore the queue reference

            assert split in {'train', 'val', 'test'}, f"Unknown split '{split}'"
            self.config = config
            self.split = split
            self.vocab = vocab
            # List of annotation dictionaries loaded from JSON files
            self.annotations: List[Dict[str, Any]] = []
            
            # Initialize L2 LMDB cache (global, shared across workers)
            self.l2_cache = None
            if config.l2_cache_enabled:
                try:
                    # Don't open LMDB in main process if using workers
                    # Each worker will open its own environment
                    if hasattr(config, 'num_workers') and config.num_workers > 0:
                        # Just store config, workers will create their own
                        self._l2_cache_config = {
                            'path': config.l2_cache_path,
                            'max_size_gb': config.l2_max_size_gb,
                            'max_readers': config.l2_max_readers,
                            'readonly': (split != 'train')
                        }
                        self.l2_cache = None
                    else:
                        # Single process mode - open directly
                        self.l2_cache = LMDBCache(
                            path=config.l2_cache_path,
                            max_size_gb=config.l2_max_size_gb,
                            max_readers=config.l2_max_readers,
                            readonly=(split != 'train')  # Only training can write
                        )
                        logger.info(f"L2 LMDB cache initialized at {config.l2_cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to initialize L2 cache: {e}")
                    self.l2_cache = None
            
            # L1 cache - much smaller per-worker cache
            # Initialize augmentation stats
            self.aug_stats = AugmentationStats() if config.collect_augmentation_stats else None
            self.stats_queue = config.stats_queue  # Queue for sending stats to main process
            self.stats_batch_interval = 10  # Send stats every N batches  
            self.batch_counter = 0
            self.color_jitter_transform = None  # Will be set in _setup_augmentation
            
            self.l1_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
            self.l1_cache_bytes = 0  # Track actual bytes in L1
            self.l1_max_size_mb = config.l1_per_worker_mb
            self.l1_min_size_mb = 32  # Minimum L1 size when shrinking

            # Ensure orientation_map_path is a Path object if provided
            self._orientation_stats = {'flips': 0, 'skipped': 0, 'processed': 0}           
            if config.orientation_map_path and isinstance(config.orientation_map_path, str):
                config.orientation_map_path = Path(config.orientation_map_path)
            # RSS tracking for stability monitoring
            self.epoch_rss_history = []
            self.current_epoch = 0
            # Initialize orientation handler for flip augmentation
            # Only needed for training split when flips are enabled
            if split == 'train' and config.random_flip_prob > 0:
                self.orientation_handler = OrientationHandler(
                    mapping_file=config.orientation_map_path,
                    random_flip_prob=config.random_flip_prob,
                    strict_mode=config.strict_orientation_validation,
                    safety_mode=config.orientation_safety_mode,
                    skip_unmapped=config.skip_unmapped
                )

                # Pre-compute mappings if vocabulary is available for better performance
                if vocab and hasattr(vocab, 'tag_to_index'):
                    all_tags = set(vocab.tag_to_index.keys())
                    self.precomputed_mappings = self.orientation_handler.precompute_all_mappings(all_tags)

                    # Validate mappings and log any issues
                    validation_issues = self.orientation_handler.validate_dataset_tags(all_tags)
                    if validation_issues:
                        # Changed: More informative warning
                        num_unmapped = len(validation_issues.get('unmapped_orientation_tags', []))
                        logger.warning(
                            f"Found {num_unmapped} orientation tags without mappings. "
                            f"These images will be skipped for flipping in conservative mode."
                        )

                        # Save validation report for review
                        from pathlib import Path as _Path  # avoid namespace confusion in compiled docs
                        validation_report_path = _Path("orientation_validation_report.json")
                        with open(validation_report_path, 'w') as f:
                            json.dump(validation_issues, f, indent=2)
                        logger.info(f"Saved validation report to {validation_report_path}")

                        # Changed: Only fail if strict mode AND issues are truly critical
                        if hasattr(config, 'strict_orientation_validation') and config.strict_orientation_validation:
                            # Don't fail for unmapped tags in conservative mode
                            if not config.skip_unmapped:
                                raise ValueError(
                                    f"Critical: Found {num_unmapped} unmapped orientation tags "
                                    f"but skip_unmapped=False. Either set skip_unmapped=True "
                                    f"or add mappings. Check {validation_report_path}"
                                )
                else:
                    self.precomputed_mappings = None

                # Initialize monitor for training
                self.orientation_monitor = OrientationMonitor(threshold_unmapped=20)
            else:
                self.orientation_handler = None
                self.precomputed_mappings = None
                self.orientation_monitor = None
            # Set up augmentation and normalisation
            self.augmentation = self._setup_augmentation() if config.augmentation_enabled and split == 'train' else None
            self.normalize = T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
            # Determine maximum cache size in number of images
            bytes_per_element = {
                'float32': 4,
                'float16': 2,
                'uint8': 1,
            }[config.cache_precision]
            # Approximate bytes per cached image by assuming they will be stored at config.image_size
            # Use L1 cache size from config instead of full cache
            bytes_per_image = 3 * config.image_size * config.image_size * bytes_per_element
            self.l1_max_cache_items = int((config.l1_per_worker_mb * (1024 ** 2)) / bytes_per_image)
            self.cache_precision = config.cache_precision
            self.cache_lock = threading.Lock()  # Add thread lock for cache operations
            self.cache_stats = {'l1_hits': 0, 'l2_hits': 0, 'disk_loads': 0}

            # Initialise locks and counters for error handling.  `_error_counts` is
            # used to track temporary I/O failures per image and must be
            # accessed under `_error_counts_lock` to ensure thread safety.
            self._error_counts_lock = threading.Lock()
            self._error_counts: Dict[str, int] = {}
            # Track permanently failed images
            self._failed_images: set[str] = set()

            # Initialize known-good sample pool for fallback
            self._known_good_indices: List[int] = []
            self._known_good_lock = threading.Lock()
            self._populate_known_good_pool()

            # Max retry attempts - scale with dataset size
            self._max_retry_attempts = min(16, max(8, len(self.annotations) // 1000))
            logger.info(f"Max retry attempts set to {self._max_retry_attempts} for {len(self.annotations)} samples")

            # Initialize validation index for lazy validation
            self.validation_index = None
            if split == 'train':
                try:
                    validation_index_base = Path(os.environ.get('OPPAI_VALIDATION_DIR', './validation'))
                    validation_index_path = validation_index_base / "index.sqlite"
                    self.validation_index = ValidationIndex(validation_index_path)
                    self.background_validator = BackgroundValidator(self.validation_index, config.data_dir)
                    self.background_validator.start()
                    logger.info(f"Validation index initialized at {validation_index_path}")
                except Exception as e:
                    logger.warning(f"Failed to initialize validation index: {e}")
                    # Fall back to eager validation if index fails
                    if config.validate_on_init:
                        self._validate_dataset_images()

            # Filter out known bad images if skip_error_samples is enabled
            self._filter_failed_images()            

            # Load annotation metadata from the provided JSON files.  This
            # populates ``self.annotations`` with valid entries.
            self._load_annotations(json_files)
            # Initialize working set and sampling attributes
            self.working_set_sampler = None
            self.epoch_indices = None
            self.sample_weights = None
            self.working_set_weights = None

            # Compute sampling weights if frequency‑weighted sampling is
            # enabled and this is the training split; otherwise, assign
            # ``None`` so that standard shuffling is used.
            if self.config.frequency_weighted_sampling and split == 'train':
                self._calculate_sample_weights()
                # Initialize working set sampler if enabled
                if config.use_working_set_sampler:
                    self.working_set_sampler = WorkingSetSampler(
                        dataset_size=len(self.annotations),
                        working_set_pct=config.working_set_pct,
                        max_items=config.working_set_max_items,
                        trickle_pct=config.trickle_in_pct,
                        max_new_per_epoch=config.max_new_uniques_per_epoch,
                        refresh_epochs=config.working_set_refresh_epochs,
                        sample_weights=self.sample_weights
                    )
                    # Get initial epoch indices
                    self.epoch_indices = self.working_set_sampler.get_epoch_indices()
                    # Recompute weights for the working set only
                    self._calculate_working_set_weights()
                    logger.info(
                        f"Working set sampler active with {len(self.epoch_indices)} indices for epoch"
                    )

            effective_size = len(self.epoch_indices) if self.epoch_indices else len(self.annotations)
            logger.info(f"Dataset initialised with {len(self.annotations)} total samples, "
                       f"{effective_size} effective samples for split '{split}'")

            # Monitor memory at initialization
            self._log_memory_status("Dataset initialization")

    def _populate_known_good_pool(self, sample_size: int = 100):
        """Populate a pool of known-good sample indices for fallback"""
        if len(self.annotations) == 0:
            return

        sample_size = min(sample_size, len(self.annotations))
        test_indices = np.random.choice(len(self.annotations), size=sample_size, replace=False)

        good_indices = []
        for idx in test_indices:
            image_path = self.annotations[idx]['image_path']
            try:
                # Quick validation - just try to open
                with Image.open(image_path) as img:
                    _ = img.size
                good_indices.append(idx)
            except Exception:
                continue

        with self._known_good_lock:
            self._known_good_indices = good_indices

        logger.info(f"Known-good pool populated with {len(good_indices)}/{sample_size} valid samples")

    def _load_annotations(self, json_files: List[Path]) -> None:
        """Parse annotation files and populate ``self.annotations``.

        Images with at least one tag are kept.  Unknown tags are retained
        (they will map to ``<UNK>`` when encoding).  Missing or invalid
        entries are skipped silently but logged.
        """
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Normalize to list format: single dict → [dict], invalid → skip
                if isinstance(data, dict):
                    # Single annotation object - wrap in list
                    data = [data]
                elif not isinstance(data, list):
                    # Neither dict nor list - invalid format
                    logger.warning(
                        f"Skipping {json_file}: expected list or dict, got {type(data).__name__}"
                    )
                    continue

                for entry in data:
                    # Validate entry is a dict before accessing keys
                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping non-dict entry in {json_file}: {type(entry).__name__}")
                        continue

                    filename = entry.get('filename')
                    tags_field = entry.get('tags')
                    if not filename or not tags_field:
                        continue
                    tags_list: List[str]
                    if isinstance(tags_field, str):
                        tags_list = tags_field.split()
                    elif isinstance(tags_field, list):
                        tags_list = tags_field
                    else:
                        continue
                    # Deduplicate tags while preserving order
                    seen = set()
                    deduplicated_tags = []
                    for tag in tags_list:
                        if tag and tag not in seen:  # Also filter out empty strings
                            seen.add(tag)
                            deduplicated_tags.append(tag)

                    record: Dict[str, Any] = {
                        'image_path': str(json_file.parent / filename),
                        'tags': deduplicated_tags,
                        'rating': entry.get('rating', 'unknown'),
                        'num_tags': len(deduplicated_tags)
                    }
                    self.annotations.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")

    def _validate_dataset_images(self) -> None:
        """Validate that all images in the dataset are readable.
        
        This runs at initialization to identify problematic images early
        rather than discovering them during training.
        """
        logger.info("Validating dataset images...")
        failed_count = 0
        total = len(self.annotations)
        
        for i, anno in enumerate(self.annotations):
            if i % 1000 == 0:
                logger.info(f"Validated {i}/{total} images...")
            
            image_path = anno['image_path']
            try:
                # Quick validation - just try to open and get size
                with Image.open(image_path) as img:
                    _ = img.size
            except Exception as e:
                logger.warning(f"Image validation failed for {image_path}: {e}")
                self._failed_images.add(image_path)
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(
                f"Found {failed_count}/{total} ({failed_count/total*100:.1f}%) unreadable images. "
                f"These will be {'skipped' if self.config.skip_error_samples else 'replaced with blank samples'}."
            )
            # Save report of failed images
            report_path = Path("failed_images_report.txt")
            with open(report_path, 'w') as f:
                for path in sorted(self._failed_images):
                    f.write(f"{path}\n")
            logger.info(f"Saved list of failed images to {report_path}")

    def _filter_failed_images(self) -> None:
        """Remove known failed images from annotations if skip_error_samples is enabled."""
        if self.config.skip_error_samples and self._failed_images:
            original_count = len(self.annotations)
            self.annotations = [a for a in self.annotations if a['image_path'] not in self._failed_images]
            filtered_count = original_count - len(self.annotations)
            logger.info(f"Filtered out {filtered_count} samples with known bad images")

    def _calculate_sample_weights(self) -> None:
        """Compute sampling weights for frequency‑weighted sampling."""
        weights: List[float] = []
        # Build a set of tags that influence orientation oversampling
        orientation_tags = set()
        if self.orientation_handler is not None and self.precomputed_mappings:
            orientation_tags.update(self.orientation_handler.explicit_mappings.keys())
            orientation_tags.update(self.orientation_handler.reverse_mappings.keys())
        for anno in self.annotations:
            w = 0.0
            has_orientation_tag = False 
            for tag in anno['tags']:
                freq = self.vocab.tag_frequencies.get(tag, 1)
                # Inverse frequency weighting
                w += (1.0 / max(freq, 1)) ** self.config.sample_weight_power
            # Average over number of tags to avoid biasing multi‑tag images
            w = w / max(1, len(anno['tags']))
            weights.append(w)
        weights_arr = np.array(weights, dtype=np.float64)
        if weights_arr.size == 0:
            weights_arr = np.ones(len(self.annotations), dtype=np.float64)
        # Normalise weights to sum to one
        weights_arr = weights_arr / weights_arr.sum() if weights_arr.sum() > 0 else weights_arr
        self.sample_weights = weights_arr
        logger.info(
            f"Sample weights calculated (min={weights_arr.min():.6f}, max={weights_arr.max():.6f})"
        )

    def _calculate_working_set_weights(self) -> None:
        """Compute sampling weights for the current working set only."""
        if self.epoch_indices is None or self.sample_weights is None:
            return

        working_weights: List[float] = []
        for idx in self.epoch_indices:
            if idx < len(self.sample_weights):
                working_weights.append(self.sample_weights[idx])
            else:
                # Fallback uniform weight if index out of range
                working_weights.append(1.0 / len(self.annotations))

        self.working_set_weights = np.array(working_weights, dtype=np.float64)
        if self.working_set_weights.size > 0:
            self.working_set_weights = self.working_set_weights / self.working_set_weights.sum()
            logger.info(
                f"Working set weights calculated for {len(self.working_set_weights)} samples "
                f"(min={self.working_set_weights.min():.6f}, max={self.working_set_weights.max():.6f})"
            )
        else:
            self.working_set_weights = None

    def _setup_augmentation(self) -> Optional[T.Compose]:
        """Create an augmentation pipeline for the training split.

        Carefully tuned for anime images to preserve colour accuracy while
        providing useful variation.  The pipeline excludes horizontal flips,
        which are handled explicitly in :meth:`__getitem__` to enable
        orientation‑aware tag remapping. Also excludes spatial transforms
        which should be applied before letterbox padding.
        """
        transforms: List[Any] = []

        # Colour jitter with conservative values
        if self.config.color_jitter:
            # Use tracked version for stats collection
            self.color_jitter_transform = TrackedColorJitter(
                brightness=self.config.color_jitter_brightness,
                contrast=self.config.color_jitter_contrast,
                saturation=self.config.color_jitter_saturation,
                hue=self.config.color_jitter_hue,
            )
            transforms.append(self.color_jitter_transform)

        # Random gamma correction with clamping
        def gamma_transform(img: torch.Tensor) -> torch.Tensor:
            # Reduced range: 0.9 to 1.1 for anime images
            gamma = float(np.random.uniform(0.9, 1.1))
            result = TF.adjust_gamma(img, gamma=gamma)
            # Clamp to avoid out-of-range values
            return torch.clamp(result, 0.0, 1.0)

        transforms.append(T.Lambda(gamma_transform))

        # RandAugment
        if self.config.randaugment_num_ops > 0 and self.config.randaugment_magnitude > 0:
            transforms.append(v2.RandAugment(num_ops=self.config.randaugment_num_ops, magnitude=self.config.randaugment_magnitude))

        # Final safety clamp to ensure valid range
        transforms.append(T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)))

        # Random Erasing (applied after all other transforms)
        if self.config.random_erasing_p > 0:
            transforms.append(v2.RandomErasing(p=self.config.random_erasing_p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))


        return T.Compose(transforms) if transforms else None

    def __len__(self) -> int:
        # Use working set size if active
        if self.epoch_indices is not None:
            return len(self.epoch_indices)
        return len(self.annotations)

    def _get_actual_index(self, idx: int) -> int:
        """Map dataset index to actual annotation index with bounds checking"""
        if self.epoch_indices:
            size = len(self.epoch_indices)
            if idx >= size or idx < -size:
                logger.warning(
                    f"Index {idx} out of bounds for epoch_indices (size {size}), using modulo"
                )
                idx %= size
            return self.epoch_indices[idx]

        total = len(self.annotations)
        if idx >= total or idx < -total:
            logger.warning(
                f"Index {idx} out of bounds for annotations (size {total}), using modulo"
            )
            idx %= total
        return idx
    
    def _log_memory_status(self, context: str = ""):
        """Log current memory status"""
        mem = psutil.virtual_memory()
        free_pct = (mem.available / mem.total) * 100
        
        if free_pct < self.config.critical_free_ram_pct:
            logger.warning(f"CRITICAL memory pressure at {context}: {free_pct:.1f}% free")
        elif free_pct < self.config.low_free_ram_pct:
            logger.info(f"Low memory at {context}: {free_pct:.1f}% free")
    
    def _check_memory_pressure(self) -> str:
        """Check current memory pressure level"""
        mem = psutil.virtual_memory()
        free_pct = (mem.available / mem.total) * 100
        
        if free_pct < self.config.critical_free_ram_pct:
            return "critical"
        elif free_pct < self.config.low_free_ram_pct:
            return "low"
        elif free_pct < self.config.high_free_ram_pct:
            return "high"
        return "normal"

    def _load_image(self, image_path: str) -> Tuple[torch.Tensor, bool]:
        """Load an image using tiered cache (L1 -> L2 -> disk)"""

        # Lazy initialization of L2 cache for workers
        if self.l2_cache is None and hasattr(self, '_l2_cache_config'):
            try:
                self.l2_cache = LMDBCache(**self._l2_cache_config)
                logger.debug(f"L2 cache initialized in worker process {os.getpid()}")
            except Exception as e:
                logger.warning(f"Failed to initialize L2 cache in worker: {e}")
                self.l2_cache = None
                if hasattr(self, '_l2_cache_config'):
                    delattr(self, '_l2_cache_config')

        # Generate deduplicated cache key with transform signature
        transform_sig = f"size{self.config.image_size}_norm{hash(self.config.normalize_mean)}"
        cache_key = _compute_cache_key(
            image_path,
            transform_signature=transform_sig,
            target_shape=(3, self.config.image_size, self.config.image_size),
            dtype=self.config.canonical_cache_dtype
        )
        
        # Check L1 cache first (per-worker, tiny)
        with self.cache_lock:
            if image_path in self.l1_cache:
                self.cache_stats['l1_hits'] += 1
                cached_data = self.l1_cache[image_path]
                self.l1_cache.move_to_end(image_path)  # LRU
                
                if isinstance(cached_data, tuple):
                    cached, was_composited = cached_data
                else:
                    cached = cached_data
                    was_composited = False

                if cached.dtype == torch.uint8:
                    result = cached.float() / 255.0
                elif cached.dtype == torch.float16:
                    result = cached.float()
                else:
                    result = cached.clone()

                return result, was_composited
        # Check L2 cache (LMDB, global)
        if self.l2_cache is not None:
            cached = self.l2_cache.get(cache_key)
            if cached is not None:
                self.cache_stats['l2_hits'] += 1
                # Deserialize and reshape from flattened tensor
                was_composited = False  # TODO: store this metadata

                # The cached tensor is flattened, need to reshape to (C, H, W)
                # We know it's stored as uint8 and the target shape
                if cached.dim() == 1:
                    # Calculate expected shape
                    C, H, W = 3, self.config.image_size, self.config.image_size
                    expected_elements = C * H * W
                    if cached.numel() >= expected_elements:
                        # Reshape to image dimensions
                        cached = cached[:expected_elements].view(C, H, W)
                    else:
                        # Size mismatch, load from disk instead
                        logger.warning(f"Cached tensor size mismatch for {image_path}, loading from disk")
                        cached = None

                if cached is not None:
                    tensor = cached.float() / 255.0  # Convert uint8 to float

                    # Add to L1 with memory-aware eviction
                    self._add_to_l1_cache(image_path, (tensor.clone(), was_composited))

                    return tensor, was_composited
        
        # Lazy validation before loading from disk
        if self.validation_index is not None:
            status = self.validation_index.get_status(image_path)
            if status == 'failed':
                # Check if we should retry
                if not self.validation_index.should_retry(image_path):
                    # Skip permanently failed image
                    logger.debug(f"Skipping permanently failed image: {image_path}")
                    return torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32), False
        # Load from disk
        self.cache_stats['disk_loads'] += 1
        
        # Helper function to resolve pad color
        def _resolve_pad_color(cfg, default=(128, 128, 128)):
            val = getattr(cfg, "pad_color", None)
            if val is None and isinstance(cfg, dict):
                val = cfg.get("pad_color", None)
            if val is None:
                return default
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                return tuple(int(v) for v in val[:3])
            return default
        
        try:
            # Open without forcing RGB so we can properly handle alpha first
            with Image.open(image_path) as pil_img:
                was_composited = False

                # Validate and update index
                if self.validation_index is not None:
                    self.validation_index.set_status(image_path, 'valid')

                pad_color = _resolve_pad_color(self.config)

                # Composite transparent images
                if pil_img.mode in ('RGBA', 'LA') or ('transparency' in pil_img.info):
                    was_composited = True
                    rgba = pil_img.convert('RGBA')
                    bg = Image.new('RGB', rgba.size, pad_color)
                    bg.paste(rgba, mask=rgba.split()[3])
                    pil_img = bg
                else:
                    pil_img = pil_img.convert('RGB')

                tensor = TF.to_tensor(pil_img)

            # Add to L2 cache if memory allows
            if self.l2_cache is not None:
                memory_pressure = self._check_memory_pressure()
                if memory_pressure in ["normal", "high"]:
                    # Store as uint8 to save space
                    cached_tensor = (tensor * 255).to(torch.uint8)
                    self.l2_cache.put(cache_key, cached_tensor, check_memory=True)
            
            # Add to L1 cache
            self._add_to_l1_cache(image_path, (tensor.clone(), was_composited))
            
            return tensor, was_composited
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Update validation index
            if self.validation_index is not None:
                self.validation_index.set_status(image_path, 'failed', str(e))
            # Return black image
            return torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32), False
    
    def _add_to_l1_cache(self, key: str, value: Tuple[torch.Tensor, bool]):
        """Add item to L1 cache with memory-aware eviction"""
        with self.cache_lock:
            # Check memory pressure
            memory_pressure = self._check_memory_pressure()
            
            if memory_pressure == "critical":
                # Don't add to cache under critical pressure
                logger.debug(f"Critical memory: skipping L1 cache add")
                return
            elif memory_pressure == "low":
                # Reduce L1 size under low memory
                target_mb = max(self.l1_min_size_mb, self.l1_max_size_mb // 2)
                max_bytes = target_mb * 1024 * 1024
            elif memory_pressure == "high":
                # Slightly reduce L1 under high memory
                target_mb = self.l1_max_size_mb * 0.75
                max_bytes = target_mb * 1024 * 1024
            else:
                max_bytes = self.l1_max_size_mb * 1024 * 1024
            
            # Evict if needed
            tensor, was_composited = value
            
            # Calculate actual bytes for this tensor
            tensor_bytes = tensor.element_size() * tensor.numel()
            
            # Evict until we have space
            while self.l1_cache_bytes + tensor_bytes > max_bytes and len(self.l1_cache) > 0:
                evicted_key, evicted_value = self.l1_cache.popitem(last=False)  # Remove LRU
                if isinstance(evicted_value, tuple):
                    evicted_tensor = evicted_value[0]
                else:
                    evicted_tensor = evicted_value
                self.l1_cache_bytes -= evicted_tensor.element_size() * evicted_tensor.numel()
            
            # Store compressed version
            if self.cache_precision == 'uint8':
                cached = (tensor * 255).to(torch.uint8)
            elif self.cache_precision == 'float16':
                cached = tensor.half()
            else:
                cached = tensor.clone()
            
            self.l1_cache[key] = (cached, was_composited)
            self.l1_cache_bytes += cached.element_size() * cached.numel()

    
    def new_epoch(self):
        """Called at the start of a new epoch to update working set"""
        self.current_epoch += 1
        
        # Track RSS for stability monitoring
        mem = psutil.Process().memory_info()
        current_rss_gb = mem.rss / (1024**3)
        self.epoch_rss_history.append(current_rss_gb)
        
        # Check RSS stability over last 3 epochs
        if len(self.epoch_rss_history) >= 3:
            recent_rss = self.epoch_rss_history[-3:]
            rss_variation = (max(recent_rss) - min(recent_rss)) / max(recent_rss) * 100
            
            if rss_variation > 10:
                logger.warning(f"RSS variation over last 3 epochs: {rss_variation:.1f}% (target <=10%)")
            else:
                logger.info(f"RSS stable: {rss_variation:.1f}% variation over last 3 epochs")
  
        if self.working_set_sampler is not None:
            self.epoch_indices = self.working_set_sampler.get_epoch_indices()
            logger.info(f"New epoch: working set updated to {len(self.epoch_indices)} indices")
            # Recompute weights for new working set
            self._calculate_working_set_weights()

            # Track unique IDs
            unique_ratio = len(set(self.epoch_indices)) / len(self.epoch_indices)
            logger.info(f"Unique sample ratio: {unique_ratio:.1%}")
        
        # Log cache statistics
        if hasattr(self, 'cache_stats'):
            total = sum(self.cache_stats.values())
            if total > 0:
                logger.info(
                    f"Cache stats - L1 hits: {self.cache_stats['l1_hits']/total:.1%}, "
                    f"L2 hits: {self.cache_stats['l2_hits']/total:.1%}, "
                    f"Disk loads: {self.cache_stats['disk_loads']/total:.1%}"
                )
                # Reset stats for new epoch
                self.cache_stats = {'l1_hits': 0, 'l2_hits': 0, 'disk_loads': 0}
        
        # Log L2 cache stats if available
        if self.l2_cache is not None:
            stats = self.l2_cache.get_stats()
            logger.info(
                f"L2 cache - Hit rate: {stats['hit_rate']:.1%}, "
                f"Entries: {stats['entries']}, "
                f"Size: {stats['size_bytes'] / (1024**3):.2f} GB, "
                f"Duplicate rate: {stats['duplicate_rate']:.1%}"
            )

            # Check L2 size within cap
            size_pct = (stats['size_bytes'] / (self.config.l2_max_size_gb * 1024**3)) * 100
            if size_pct > 103:  # 3% tolerance
                logger.warning(f"L2 cache size {size_pct:.1f}% of cap (target <=103%)")        

        # Check memory status
        self._log_memory_status("Epoch start")


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch an item for training or validation.

        Applies a random horizontal flip with orientation‑aware tag remapping
        for the training split.  Letterbox resizing is performed to
        preserve aspect ratio.  Augmentation and normalisation are then
        applied."""

        # Iterative retry logic with visited tracking
        visited_indices = set()
        attempts = 0
        original_idx = idx

        while attempts < self._max_retry_attempts:
            attempts += 1

            # Avoid infinite loops by checking visited indices
            if idx in visited_indices:
                # Try to get a known-good sample
                with self._known_good_lock:
                    if self._known_good_indices:
                        idx = random.choice(self._known_good_indices)
                        if idx in visited_indices:
                            # Even our good samples have been tried, give up
                            break
                    else:
                        # No good samples available, increment and continue
                        idx = (idx + 1) % len(self.annotations)

            visited_indices.add(idx)

            # Map index through working set if active
            actual_idx = self._get_actual_index(idx)

            if actual_idx < 0 or actual_idx >= len(self.annotations):
                # Invalid index, try next
                idx = (idx + 1) % len(self.annotations)
                continue

            # Use actual_idx instead of idx when accessing annotations
            anno = self.annotations[actual_idx]
            image_path = anno['image_path']

            # Skip if this is a known failed image
            if image_path in self._failed_images:
                idx = (idx + 1) % len(self.annotations)
                continue

            # Track error count for this specific image
            with self._error_counts_lock:
                error_count = self._error_counts.get(image_path, 0)

            try:
                # Load image tensor and get composite flag
                image, was_composited = self._load_image(image_path)

                # Validate image tensor
                if image is None or torch.isnan(image).any() or torch.isinf(image).any():
                    raise ValueError(f"Invalid image tensor for {image_path}")

                # Copy tags so we can mutate without altering the original
                tags = list(anno['tags'])

                # Update tags if image was composited from transparency
                if was_composited:
                    # Remove transparent_background if present
                    tags = [t for t in tags if t != 'transparent_background']
                    # Add gray_background if not already present
                    if 'gray_background' not in tags:
                        tags.append('gray_background')

                was_flipped = False  # Track if image was flipped
                # Random horizontal flip with orientation-aware tag swapping
                if (
                    self.orientation_handler is not None
                    and self.split == 'train'
                    and np.random.rand() < self.config.random_flip_prob
                ):
                    swapped_tags, should_flip = self.orientation_handler.handle_complex_tags(tags)

                    # Track flip stats
                    if self.aug_stats:
                        self.aug_stats.flip_total += 1

                    if should_flip:
                        image = TF.hflip(image)
                        was_flipped = True
                        self._orientation_stats['flips'] += 1
                        if self.aug_stats:
                            self.aug_stats.flip_safe += 1
                    else:
                        self._orientation_stats['skipped'] += 1
                        # Determine skip reason from handler stats
                        if self.aug_stats and self.orientation_handler:
                            handler_stats = self.orientation_handler.stats
                            if handler_stats.get('blocked_by_text', 0) > 0:
                                self.aug_stats.flip_skipped_text += 1
                            elif handler_stats.get('blocked_by_safety', 0) > 0:
                                self.aug_stats.flip_blocked_safety += 1
                            else:
                                self.aug_stats.flip_skipped_unmapped += 1

                    self._orientation_stats['processed'] += 1
                    tags = swapped_tags
                    if self.orientation_monitor:
                        self.orientation_monitor.check_health(self.orientation_handler)

                    # Send orientation stats to queue periodically (similar to aug stats)
                    if (self.stats_queue and 
                        self._orientation_stats['processed'] % self.stats_batch_interval == 0):
                        try:
                            stats_copy = {
                                'flips': self._orientation_stats.get('flips', 0),
                                'skipped': self._orientation_stats.get('skipped', 0),
                                'processed': self._orientation_stats.get('processed', 0)
                            }
                            self.stats_queue.put_nowait(('orientation_stats', stats_copy))
                            # Reset local counters after sending
                            self._orientation_stats = {'flips': 0, 'skipped': 0, 'processed': 0}
                        except queue.Full:
                            pass  # Queue full, skip this batch

                # Track color jitter if applied
                if self.augmentation is not None and self.split == 'train':
                    if self.color_jitter_transform and hasattr(self.color_jitter_transform, 'last_params'):
                        if self.aug_stats:
                            self.aug_stats.jitter_applied += 1
                            params = self.color_jitter_transform.last_params
                            # Keep only last N samples to avoid unbounded growth
                            if len(self.aug_stats.jitter_brightness_factors) < 1000:
                                self.aug_stats.jitter_brightness_factors.append(params.get('brightness', 1.0))
                                self.aug_stats.jitter_contrast_factors.append(params.get('contrast', 1.0))

                # Ensure image tensor is properly shaped before any transforms
                if image.dim() != 3:
                    logger.warning(f"Image tensor has unexpected dimensions: {image.shape} for {image_path}")
                    # Try to reshape if it's flattened
                    if image.dim() == 1:
                        try:
                            image = image.view(3, self.config.image_size, self.config.image_size)
                        except RuntimeError:
                            # Can't reshape, treat as error
                            raise ValueError(f"Cannot reshape tensor for {image_path}")
                    else:
                        raise ValueError(f"Invalid tensor dimensions for {image_path}")

                # Apply color augmentations BEFORE letterbox/padding
                # This ensures padding areas remain consistent
                if self.augmentation is not None and self.split == 'train':
                    try:
                        image = self.augmentation(image)

                        # Ensure augmented values are valid
                        if torch.isnan(image).any() or torch.isinf(image).any():
                            logger.warning(
                                f"NaN/Inf detected after augmentation for {image_path}, skipping augmentation"
                            )
                            # Reload clean image
                            image, _ = self._load_image(image_path)
                            # Re-apply flip if it was done
                            if was_flipped:
                                image = TF.hflip(image)
                        elif (image < 0).any() or (image > 1).any():
                            logger.debug(
                                f"Values outside [0,1] detected, clamping for {image_path}"
                            )
                            image = torch.clamp(image, 0.0, 1.0)
                    except Exception as e:
                        logger.warning(
                            f"Augmentation failed for {image_path}: {e}, skipping augmentation"
                        )

                # Track crop stats
                crop_scale = 1.0
                crop_aspect = 1.0

                # Apply RandomResizedCrop BEFORE letterbox if configured
                if (self.augmentation is not None and
                    self.split == 'train' and
                    self.config.random_crop_scale != (1.0, 1.0)):
                    try:
                        # Track crop stats
                        crop_scale = random.uniform(*self.config.random_crop_scale)
                        crop_aspect = random.uniform(0.9, 1.1)  # Default aspect range

                        if self.aug_stats:
                            self.aug_stats.crop_applied += 1
                            if len(self.aug_stats.crop_scales) < 1000:
                                self.aug_stats.crop_scales.append(crop_scale)
                                self.aug_stats.crop_aspects.append(crop_aspect)
                        image = T.RandomResizedCrop(
                            self.config.image_size,
                            scale=self.config.random_crop_scale,
                            ratio=(0.9, 1.1),
                            interpolation=T.InterpolationMode.BICUBIC
                        )(image)
                        lb_info = {'scale': 1.0, 'pad': (0, 0, 0, 0)}
                    except Exception as e:
                        logger.warning(f"RandomResizedCrop failed for {image_path}: {e}, using letterbox instead")
                        image, lb_info = letterbox_resize(
                            image,
                            target_size=self.config.image_size,
                            pad_color=self.config.pad_color,
                            patch_size=self.config.patch_size,
                        )
                else:
                    # Perform letterbox resize to preserve aspect ratio
                    image, lb_info = letterbox_resize(
                        image,
                        target_size=self.config.image_size,
                        pad_color=self.config.pad_color,
                        patch_size=self.config.patch_size,
                    )

                # Track resize stats
                if self.aug_stats and lb_info:
                    if len(self.aug_stats.resize_scales) < 1000:
                        self.aug_stats.resize_scales.append(lb_info['scale'])
                        total_pad = sum(lb_info['pad'])
                        self.aug_stats.resize_pad_pixels.append(total_pad)

                # Update batch counter and send stats if needed
                if self.aug_stats:
                    self.aug_stats.image_count += 1
                    self.batch_counter += 1
                    if self.batch_counter % self.stats_batch_interval == 0:
                        self.aug_stats.batch_count += 1
                        self._send_stats_to_queue()

                # Normalise
                if image.dtype != torch.float32:
                    image = image.float()
                image = self.normalize(image)

                # Encode tags and rating
                tag_labels = self.vocab.encode_tags(tags)
                rating_label = self.vocab.rating_to_index.get(anno['rating'], self.vocab.rating_to_index['unknown'])

                # Success! Reset error count and add to known-good pool
                with self._error_counts_lock:
                    if image_path in self._error_counts:
                        del self._error_counts[image_path]

                # Add successful index to known-good pool (with probability to avoid bias)
                if random.random() < 0.1:  # 10% chance
                    with self._known_good_lock:
                        if actual_idx not in self._known_good_indices:
                            self._known_good_indices.append(actual_idx)
                            # Keep pool size bounded
                            if len(self._known_good_indices) > 200:
                                self._known_good_indices.pop(0)

                # Package the sample as a dictionary
                return {
                    'image': image,
                    'labels': {
                        'tags': tag_labels,
                        'rating': rating_label,
                    },
                    'metadata': {
                        'index': actual_idx,
                        'path': anno['image_path'],
                        'num_tags': len(tags),
                        'tags': tags,
                        'rating': anno['rating'],
                        'scale': lb_info['scale'],
                        'pad': lb_info['pad'],
                        'was_composited': was_composited,
                        'was_flipped': was_flipped,
                    },
                }

            except (IOError, OSError, ValueError) as e:
                # Track error and mark as failed if exceeded retries
                with self._error_counts_lock:
                    self._error_counts[image_path] = error_count + 1
                    if error_count >= 3:
                        self._failed_images.add(image_path)

                # Log only first occurrence per image
                if error_count == 0:
                    logger.warning(f"Error loading {image_path}: {e}")

                # Try next index
                idx = (idx + 1) % len(self.annotations)
                continue

            except Exception as e:
                # Unexpected error - log once and mark as failed
                if image_path not in self._failed_images:
                    logger.error(f"Unexpected error loading {image_path}: {e}")
                    self._failed_images.add(image_path)

                # Try next index
                idx = (idx + 1) % len(self.annotations)
                continue

        # Exhausted all attempts - return error sample
        logger.warning(f"Failed to load sample after {attempts} attempts, returning error sample for index {original_idx}")
        return self._create_error_sample(original_idx, "exhausted_retries", "max_retries_exceeded")
        
    def _send_stats_to_queue(self):
        """Send augmentation stats to main process via queue."""
        if self.stats_queue and self.aug_stats:
            try:
                # Send a copy of current stats
                stats_copy = copy.deepcopy(self.aug_stats)
                self.stats_queue.put_nowait(('aug_stats', stats_copy))
                # Reset local stats after sending
                self.aug_stats = AugmentationStats()
            except queue.Full:
                pass  # Queue full, skip this batch
    def _create_error_sample(self, idx: int, image_path: str, error_type: str) -> Dict[str, Any]:
        """Create an error sample with appropriate defaults.

        This should only be used for recoverable errors like temporary I/O issues.
        """
        # Use letterbox padding on a zero image so downstream patchify has correct shape.
        blank = torch.zeros(3, self.config.image_size, self.config.image_size, dtype=torch.float32)
        padded, lb_info = letterbox_resize(
            blank,
            target_size=self.config.image_size,
            pad_color=self.config.pad_color,
            patch_size=self.config.patch_size,
        )
        # Ensure consistent tensor shapes and dtypes
        tag_labels = torch.zeros(len(self.vocab.tag_to_index), dtype=torch.float32)
        rating_label = torch.tensor(self.vocab.rating_to_index.get('unknown', 4), dtype=torch.long)

        return {
            'image': padded,
            'labels': {
                'tags': tag_labels,
                'rating': rating_label,
            },
            'metadata': {
                'index': idx,
                'path': image_path,
                'num_tags': 0,
                'tags': [],
                'rating': 'unknown',
                'error_type': error_type,
                'is_error_sample': True,  # Flag for collate_fn
                'scale': lb_info['scale'],
                'pad': lb_info['pad'],
                'was_composited': False,
            },
        }


    def get_orientation_stats(self) -> Dict[str, Any]:
        """Get orientation statistics from this dataset instance.
        
        Note: In multi-worker settings, these are local to this worker process.
        Statistics are not aggregated across workers automatically.
        
        Returns:
            Dictionary with orientation statistics
        """
        if not hasattr(self, '_orientation_stats'):
            return {
                'total_flips': 0,
                'skipped_flips': 0,
                'processed_samples': 0,
                'flip_rate': 0.0,
                'skip_rate': 0.0,
                'has_handler': False,
                'worker_local': True
            }
        
        stats = self._orientation_stats.copy()
        processed = stats.get('processed', 0)
        
        return {
            'total_flips': stats.get('flips', 0),
            'skipped_flips': stats.get('skipped', 0),
            'processed_samples': processed,
            'flip_rate': stats['flips'] / max(1, processed),
            'skip_rate': stats['skipped'] / max(1, processed),
            'has_handler': self.orientation_handler is not None,
            'worker_local': True  # Flag to indicate these are worker-local stats
        }

    def __del__(self):
        """Cleanup when dataset is destroyed."""
        if hasattr(self, 'background_validator'):
            self.background_validator.stop()

def create_dataloaders(
    data_config: DataConfig,
    validation_config: ValidationConfig,
    vocab_path: Path,
    active_data_path: Path,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    frequency_sampling: bool = True,
    seed: Optional[int] = None,
    log_queue: Optional[object] = None,
    sampler_type: str = "uniform",
    max_repeat_factor: float = 3.0,
) -> Tuple[DataLoader, DataLoader, TagVocabulary]:
    """Construct training and validation dataloaders with enhanced memory control."""
    
    logger.info(f"HDF5_loader received active data path: {active_data_path}")

    json_files = list(active_data_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {active_data_path}")

    json_files_sorted = sorted(json_files)
    np.random.shuffle(json_files_sorted)
    split_idx = int(len(json_files_sorted) * 0.9)
    train_files = json_files_sorted[:split_idx]
    val_files = json_files_sorted[split_idx:]

    vocab = TagVocabulary(vocab_path)
    if not vocab_path.exists():
        vocab.build_from_annotations(json_files_sorted, top_k=None)
        vocab.save_vocabulary(vocab_path)

    train_dataset = SimplifiedDataset(data_config, train_files, split='train', vocab=vocab)
    val_dataset = SimplifiedDataset(data_config, val_files, split='val', vocab=vocab)

    base_seed = int(seed if seed is not None else torch.initial_seed() % (2**31 - 1))
    generator = torch.Generator()
    generator.manual_seed(base_seed)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # In order to apply MixUp or CutMix, we need to wrap the default collate_fn.
    # These transforms operate on batches, not individual samples.
    train_collate_fn = collate_fn
    if data_config.mixup_alpha > 0.0 or data_config.cutmix_alpha > 0.0:
        mixup_cutmix_transforms = []
        if data_config.mixup_alpha > 0.0:
            mixup_cutmix_transforms.append(
                v2.MixUp(alpha=data_config.mixup_alpha, num_classes=len(vocab.tag_to_index))
            )
        if data_config.cutmix_alpha > 0.0:
            mixup_cutmix_transforms.append(
                v2.CutMix(alpha=data_config.cutmix_alpha, num_classes=len(vocab.tag_to_index))
            )

        mixup_cutmix = v2.RandomChoice(mixup_cutmix_transforms)

        def collate_fn_with_aug(batch):
            # The default collate_fn returns a dict. v2.MixUp/CutMix expect (images, labels)
            # We need to adapt it to our dictionary structure.
            collated_batch = collate_fn(batch)

            # The labels to be mixed are the tag_labels. The rating_labels should not be mixed.
            # We can create a simple lambda to extract the tag_labels for the transform.
            images, tag_labels = mixup_cutmix(collated_batch['images'], collated_batch['tag_labels'])

            collated_batch['images'] = images
            collated_batch['tag_labels'] = tag_labels
            return collated_batch

        train_collate_fn = collate_fn_with_aug


    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        drop_last=True,
        persistent_workers=True if data_config.num_workers > 0 else False,
        collate_fn=train_collate_fn,
        worker_init_fn=_make_worker_init_fn(base_seed, log_queue),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_config.dataloader.batch_size,
        shuffle=False,
        num_workers=validation_config.dataloader.num_workers,
        pin_memory=data_config.pin_memory,
        prefetch_factor=validation_config.dataloader.prefetch_factor,
        drop_last=False,
        persistent_workers=validation_config.dataloader.persistent_workers,
        collate_fn=collate_fn, # No mixup/cutmix on validation
        worker_init_fn=_make_worker_init_fn(base_seed, log_queue),
        generator=generator,
    )

    return train_loader, val_loader, vocab


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to assemble a batch of samples."""
    # Filter out error samples if configured
    valid_batch = [item for item in batch if not item.get('metadata', {}).get('is_error_sample', False)]
    if not valid_batch:
        # All samples were errors, use original batch
        valid_batch = batch

    images_tensor = torch.stack([item['image'] for item in valid_batch])
    # Extract nested labels.  Tag labels are stacked into a 2D tensor and
    # rating labels are collected into a 1D tensor.
    tag_labels = torch.stack([item['labels']['tags'] for item in valid_batch])
    rating_labels = torch.tensor([item['labels']['rating'] for item in valid_batch], dtype=torch.long)
    # Collate metadata lists and keep padding info for downstream usage.
    metadata = {
        'indices': [item['metadata']['index'] for item in valid_batch],
        'paths': [item['metadata']['path'] for item in valid_batch],
        'num_tags': torch.tensor([item['metadata']['num_tags'] for item in valid_batch]),
        'tags': [item['metadata']['tags'] for item in valid_batch],
        'ratings': [item['metadata']['rating'] for item in valid_batch],
        'scales': [item['metadata'].get('scale') for item in valid_batch],
        'pads': [item['metadata'].get('pad') for item in valid_batch],
    }
    # Derive a per-pixel padding mask (True=content, False=padding) so downstream
    # modules (e.g., ViT attention) can ignore padded regions.
    # NOTE: This creates masks with True=content semantics, which will be auto-detected
    # and converted by mask_utils.ensure_pixel_padding_mask() if needed
    B, C, H, W = images_tensor.shape
    padding_mask = torch.ones((B, H, W), dtype=torch.bool)
    for i, pad in enumerate(metadata['pads']):
        if pad is None:
            continue
        left, top, right, bottom = pad
        if top > 0:
            padding_mask[i, :top, :] = False
        if bottom > 0:
            padding_mask[i, H - bottom:, :] = False
        if left > 0:
            padding_mask[i, :, :left] = False
        if right > 0:
            padding_mask[i, :, W - right:] = False
    
    # Return with 'images' (plural) to match training loop expectations
    return {
        'images': images_tensor,           # Changed from 'image' to 'images'
        'tag_labels': tag_labels,          # Flattened structure to match training expectations
        'rating_labels': rating_labels,    # Flattened structure to match training expectations
        'padding_mask': padding_mask,
        'metadata': metadata,
    }
