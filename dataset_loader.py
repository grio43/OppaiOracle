"""
Dataset Loader - JSON-based On-the-Fly Data Loading Pipeline

ARCHITECTURE OVERVIEW:
======================
This module provides the ACTIVE data loading pipeline for training and validation.
It loads images and metadata on-the-fly from JSON files using two modes:

1. MANIFEST MODE:
   - Requires: train.json, val.json, images/ directory
   - Legacy format for pre-split datasets

2. SIDECAR JSON MODE (Primary):
   - Per-image JSON files alongside images (e.g., 12345.json next to 12345.jpg)
   - Scans recursively, supports shard directories
   - Auto-splits 95/5 train/val with caching

CACHING LAYERS:
- Sidecar Cache: Preprocessed tensors stored as .safetensor files alongside original images
  (no LMDB, no virtual memory issues, pooled SSD-friendly, unlimited parallel reads)

VOCABULARY:
- Built automatically by vocabulary.py:create_vocabulary_from_datasets()
- Scans all JSON files, counts tag frequencies, saves to vocabulary.json
- See vocabulary.py for details

⚠️ IMPORTANT - OBSOLETE CODE WARNING:
======================================
The file `dataset_preprocessor.py` (formerly `tag_vocabulary.py`) creates HDF5 files
(training_data.h5, tag_indices.json, splits.json) that are NOT used by this system.

HISTORY:
- August 27, 2025 (commit 6727128): Removed HDF5_loader.py (2530 lines)
- Replaced with this JSON-based loader for better flexibility
- dataset_preprocessor.py became orphaned code (no consumer)

WHY JSON-BASED IS BETTER:
- On-the-fly loading allows dynamic augmentation (flips, crops, etc.)
- Orientation-aware tag swapping during training
- No preprocessing step required
- Sidecar cache provides similar performance to HDF5
- More flexible for iterative dataset refinement

DO NOT USE dataset_preprocessor.py - it is dead code maintained only for
historical reference. If you need to rebuild data, just point train_direct.py
at your JSON files and it will handle everything automatically.

USAGE:
    from dataset_loader import create_dataloaders
    train_loader, val_loader, vocab = create_dataloaders(
        data_config=config.data,
        validation_config=config.validation,
        vocab_path=config.vocab_path,
        active_data_path=dataset_root
    )
"""

# Standard library imports
import atexit
import hashlib
import json
import logging
import shutil
import zlib

# Try to use orjson for faster JSON parsing (3-5x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
    JSON_DECODE_ERRORS = (json.JSONDecodeError, orjson.JSONDecodeError)
except ImportError:
    HAS_ORJSON = False
    JSON_DECODE_ERRORS = (json.JSONDecodeError,)

# Optional file locking for sidecar cache writes
try:
    import filelock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False

# PyArrow for zero-copy metadata cache (memory-mapped, shared across workers)
try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    HAS_PYARROW = True
except ImportError:
    pa = None
    pa_ipc = None
    HAS_PYARROW = False

import logging.handlers
import multiprocessing as mp
import os
import queue
import random
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Optional, List, Dict, Any, Tuple, Set

# Third-party imports
import torch
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset, get_worker_info, DataLoader as _TorchDataLoader
from torch.utils.data.distributed import DistributedSampler

# Make torchvision optional at import time; raise only when actually used.
try:
    from torchvision import transforms  # type: ignore
except (ImportError, ModuleNotFoundError):
    transforms = None  # resolved lazily

# Torchvision v2 joint transforms (optional)
try:
    from torchvision.transforms import v2 as T
    from torchvision import tv_tensors
except (ImportError, ModuleNotFoundError, AttributeError):  # keep backward compatible
    T = None
    tv_tensors = None

# Local imports
from cache_codec import get_sidecar_path, save_sidecar, load_sidecar
from utils.cache_keys import compute_cache_config_hash
from utils.cache_monitor import monitor
from utils.metadata_ingestion import parse_tags_field
# Safetensors fallback removed - Arrow is now the only metadata cache format
from utils.path_utils import sanitize_identifier, validate_image_path, resolve_and_confine
from utils.exclusion_manager import ExclusionManager
from vocabulary import load_vocabulary_for_training, TagVocabulary
from shared_vocabulary import (
    SharedVocabularyManager,
    is_shared_memory_available,
    populate_vocab_from_shared
)

# Orientation-aware flipping (optional; keeps file usable in legacy setups)
try:
    from orientation_handler import OrientationHandler, SwapResult  # noqa: F401
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    OrientationHandler = None  # type: ignore
    SwapResult = None  # type: ignore

# Pillow resampling compatibility and truncated image handling
try:  # Pillow ≥10
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:  # Pillow <10
    RESAMPLE_BILINEAR = Image.BILINEAR

# Strict mode by default: truncated/corrupt images are rejected immediately.
# Set OO_ALLOW_TRUNCATED=1 to enable lenient mode which fills missing bytes
# with gray pixels (useful for datasets with minor corruption issues).
# Most "truncated" images are missing just a few bytes at the end (< 100 bytes)
# which Pillow would fill with gray - but this can mask data quality issues.
ALLOW_TRUNCATED = bool(int(os.environ.get("OO_ALLOW_TRUNCATED", "0")))
if ALLOW_TRUNCATED:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

# Memory bounds for error tracking to prevent unbounded growth
# These limits prevent memory exhaustion during long training runs with many failures
_MAX_RETRY_COUNTS = 10000        # Max samples to track retry counts for
_MAX_FAILED_SAMPLES = 50000     # Max permanently failed samples to track
_MAX_ERROR_STATS_TAGS = 5000    # Max unique tags to track error stats for

# Exclusion file reload interval (seconds) - how often to check for new exclusions
# from other workers. Lower = faster sync, Higher = less I/O overhead
_EXCLUSION_RELOAD_INTERVAL = 120.0

# Minimal dtype mapping for cache plumbing
_DTYPE_MAP = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def _canon_dtype(s: str) -> torch.dtype:
    return _DTYPE_MAP.get(str(s).lower(), torch.bfloat16)

# CPU BFloat16 support detection (cached globally)
_CPU_BF16_SUPPORTED = None

def _is_cpu_bf16_supported() -> bool:
    """Check if CPU supports bfloat16 operations (cached).

    Returns:
        True if CPU supports BF16, False otherwise
    """
    global _CPU_BF16_SUPPORTED
    if _CPU_BF16_SUPPORTED is None:
        try:
            # Try to create and operate on BF16 tensor
            test_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
            bf16_tensor = test_tensor.to(torch.bfloat16)
            # Try a simple operation to ensure it's actually supported
            _ = bf16_tensor + bf16_tensor
            _CPU_BF16_SUPPORTED = True
            logging.getLogger(__name__).debug("CPU bfloat16 support detected")
        except (RuntimeError, NotImplementedError):
            _CPU_BF16_SUPPORTED = False
            logging.getLogger(__name__).info("CPU does not support bfloat16 operations")
    return _CPU_BF16_SUPPORTED


# Module-level cache for Normalize transforms (avoids mutable default argument antipattern)
# Key: (mean, std) tuple, Value: transforms.Normalize instance
_NORMALIZE_TRANSFORM_CACHE: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Any] = {}


def _normalize_preserve_dtype(img: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """Apply normalization while preserving the input tensor's dtype.

    torchvision.transforms.Normalize may convert bfloat16 tensors to float32 in some
    PyTorch versions. This helper ensures the original dtype is preserved.

    Uses a module-level cache to avoid recreating Normalize objects for each sample.
    Cache key is (mean, std) tuple - typically only one unique combination per training run.

    Args:
        img: Input tensor of shape (C, H, W)
        mean: Normalization mean per channel
        std: Normalization std per channel

    Returns:
        Normalized tensor with same dtype as input
    """
    original_dtype = img.dtype
    if transforms is None:
        raise ImportError("torchvision is required for normalization")

    # Cache the Normalize transform to avoid recreating it per sample
    # This provides ~5-10% speedup for large datasets
    cache_key = (mean, std)
    if cache_key not in _NORMALIZE_TRANSFORM_CACHE:
        _NORMALIZE_TRANSFORM_CACHE[cache_key] = transforms.Normalize(mean=mean, std=std)

    normalized = _NORMALIZE_TRANSFORM_CACHE[cache_key](img)
    # Ensure dtype is preserved (may be converted to float32 by Normalize)
    if normalized.dtype != original_dtype:
        normalized = normalized.to(original_dtype)
    return normalized


def process_image_cpu(
    img: Image.Image,
    target_size: int,
    pad_color: Tuple[int, int, int]
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Process PIL image on CPU: resizing, letterboxing, and padding mask generation.
    Moves heavy PIL operations to background threads when used with ImagePreFetcher.

    Args:
        img: Source PIL Image (RGB)
        target_size: Target dimension (square)
        pad_color: RGB tuple for padding

    Returns:
        (canvas, pmask): Processed PIL Image and boolean padding mask
    """
    w, h = img.size
    # Downscale-only letterbox: preserve aspect, never upscale
    ratio = min(target_size / float(w), target_size / float(h)) if (w > 0 and h > 0) else 1.0
    scale = min(1.0, ratio)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    
    resized = img.resize((max(1, nw), max(1, nh)), RESAMPLE_BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    left = (target_size - resized.size[0]) // 2
    top = (target_size - resized.size[1]) // 2
    canvas.paste(resized, (left, top))

    pmask = torch.ones(target_size, target_size, dtype=torch.bool)
    pmask[top:top + resized.size[1], left:left + resized.size[0]] = False
    
    return canvas, pmask


## moved to utils/cache_keys.py: compute_cache_config_hash


class ResumableSampler(DistributedSampler):
    """DistributedSampler with O(1) mid-epoch resume support.

    Standard DistributedSampler requires iterating through all batches to maintain
    RNG order, which takes ~17 minutes for 5000+ batches. This sampler allows
    setting a start_index to skip directly to the resume point.

    State is serializable for checkpoint embedding.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self._start_index = 0
        self._cached_indices = None

    def set_start_index(self, index: int):
        """Set the starting index for iteration (for mid-epoch resume)."""
        self._start_index = index

    def get_state(self) -> dict:
        """Get sampler state for checkpointing."""
        return {
            'epoch': self.epoch,
            'start_index': self._start_index,
            'total_size': self.total_size,
            'num_replicas': self.num_replicas,
            'rank': self.rank,
        }

    def load_state(self, state: dict):
        """Restore sampler state from checkpoint."""
        self.set_epoch(state['epoch'])
        self._start_index = state.get('start_index', 0)

    def __iter__(self):
        # Generate indices using parent's logic
        indices = list(super().__iter__())
        self._cached_indices = indices

        # Skip to start_index for mid-epoch resume
        for i in range(self._start_index, len(indices)):
            yield indices[i]

        # Reset for next epoch
        self._start_index = 0

    def __len__(self):
        base_len = super().__len__()
        return max(0, base_len - self._start_index)


# Guarded DataLoader wrapper:
# - If num_workers == 0, drop prefetch_factor and force persistent_workers=False.
#   This avoids ValueError in PyTorch when setting multiprocessing-only args with zero workers.
class DataLoader(_TorchDataLoader):  # keep public name the same
    def __init__(self, *args, **kwargs):
        num_workers = int(kwargs.get("num_workers", 0) or 0)
        if num_workers == 0:
            # Disallow multiprocessing-only knobs in single-process mode
            kwargs.pop("prefetch_factor", None)
            kwargs["persistent_workers"] = False
        super().__init__(*args, **kwargs)


# --- JSON sidecar split caching to reduce startup I/O -----------------------
_PROJ_ROOT = Path(__file__).resolve().parent
_SPLIT_CACHE_VERSION = "2.0"
_EXCLUSION_PATTERNS = ["train.json", "val.json"]  # Manifest files excluded from sidecar mode

def _compute_exclusion_hash() -> str:
    """Compute hash of file exclusion logic to detect changes."""
    exclusion_str = ",".join(sorted(_EXCLUSION_PATTERNS))
    return hashlib.sha256(exclusion_str.encode("utf-8")).hexdigest()[:16]

def _split_cache_paths(root: Path) -> tuple[Path, Path]:
    """Return cache file paths for train/val splits for a given dataset root.

    Files live under ./logs/splits/<sha1(root)>.{train|val}.txt and contain
    absolute JSON file paths, one per line.
    """
    splits_dir = _PROJ_ROOT / "logs" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return (
        splits_dir / f"{key}.train.txt",
        splits_dir / f"{key}.val.txt",
    )

def _try_load_cached_split(root: Path, seed: int = 42) -> Optional[tuple[list[Path], list[Path]]]:
    """Load cached split files with v2.1 validation.

    Args:
        root: Dataset root directory
        seed: Random seed to validate against cached seed

    Optimizations:
      - Lazy validation: only check first 10 paths instead of 100 (10x faster)
      - Parallel existence checks using ThreadPoolExecutor
      - Early return on cache hit without full validation

    v2.1 Validation (added filesystem count check):
      - Version check
      - Exclusion hash check (detects changes to manifest file filtering)
      - Cache internal consistency check (header FILE_COUNT matches path count)
      - Seed check (detects when seed changes between runs)
      - FILESYSTEM COUNT CHECK: Scans actual filesystem to detect new/deleted files
        (prevents stale cache from silently ignoring new dataset files)
      - Sample existence check (verifies subset of cached paths still exist)
    """
    logger = logging.getLogger(__name__)
    train_file, val_file = _split_cache_paths(root)
    if train_file.exists() and val_file.exists():
        try:
            # Read both files concurrently for faster I/O
            def parse_cache_file(file_path: Path) -> tuple[dict, list]:
                """Parse cache file, extracting header and paths."""
                lines = file_path.read_text(encoding="utf-8").splitlines()
                header = {}
                paths = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        # Parse header
                        if "=" in line:
                            key, value = line[1:].split("=", 1)
                            header[key.strip()] = value.strip()
                    else:
                        # Regular path line
                        paths.append(Path(line))

                return header, paths

            with ThreadPoolExecutor(max_workers=2) as executor:
                train_future = executor.submit(parse_cache_file, train_file)
                val_future = executor.submit(parse_cache_file, val_file)
                train_header, train_list = train_future.result()
                val_header, val_list = val_future.result()

            # Validate v2.0 header (use train file header as canonical)
            cache_version = train_header.get("SPLIT_CACHE_VERSION", "1.0")
            if cache_version != _SPLIT_CACHE_VERSION:
                logger.info(
                    f"Split cache version mismatch: {cache_version} != {_SPLIT_CACHE_VERSION}. "
                    "Rebuilding with current version..."
                )
                return None

            # Validate exclusion hash
            cached_hash = train_header.get("EXCLUSION_HASH", "")
            current_hash = _compute_exclusion_hash()
            if cached_hash != current_hash:
                logger.info(
                    f"Exclusion logic changed (hash: {cached_hash} != {current_hash}). "
                    "Rebuilding split cache..."
                )
                return None

            # Validate file count with 0.1% tolerance
            if "FILE_COUNT" in train_header:
                cached_count = int(train_header["FILE_COUNT"])
                actual_count = len(train_list) + len(val_list)
                tolerance = max(100, int(cached_count * 0.001))

                if abs(cached_count - actual_count) > tolerance:
                    logger.warning(
                        f"Split cache count drift: cached={cached_count}, actual={actual_count}, "
                        f"diff={abs(cached_count - actual_count)}, tolerance={tolerance} (0.1%). "
                        "Rebuilding split cache..."
                    )
                    return None

            # Validate seed to ensure split is deterministic with current seed
            cached_seed = train_header.get("SEED", "")
            if cached_seed and str(seed) != cached_seed:
                logger.info(
                    f"Split cache seed mismatch: cached={cached_seed}, current={seed}. "
                    "Rebuilding split cache with new seed..."
                )
                return None

            # CRITICAL: Validate cached count against ACTUAL filesystem count
            # This catches new files added after the cache was created
            # Without this check, new files would be silently ignored!
            if "FILE_COUNT" in train_header:
                cached_count = int(train_header["FILE_COUNT"])
                logger.debug(f"Counting current JSON files in {root} to validate split cache...")
                # Count files on filesystem (generator avoids memory overhead)
                filesystem_count = sum(
                    1 for jp in root.rglob("*.json")
                    if jp.name not in _EXCLUSION_PATTERNS
                )
                tolerance = max(100, int(max(cached_count, filesystem_count) * 0.001))  # 0.1% of larger count

                if abs(filesystem_count - cached_count) > tolerance:
                    logger.info(
                        f"Split cache stale: filesystem has {filesystem_count:,} JSON files, "
                        f"cache has {cached_count:,} (diff={abs(filesystem_count - cached_count):,}, "
                        f"tolerance={tolerance}). Rebuilding split cache..."
                    )
                    return None
                logger.debug(f"Filesystem count validated: {filesystem_count:,} files match cache")

            # Stratified sampling: check files from beginning, end, and random middle
            # This catches orphan files anywhere in the list, not just at the start
            sample_paths = []

            def stratified_sample(file_list: list, count: int) -> list:
                """Sample from beginning, end, and random middle of a list."""
                if len(file_list) <= count:
                    return list(file_list)
                samples = []
                edge_count = min(5, count // 3)
                # First N files
                samples.extend(file_list[:edge_count])
                # Last N files
                samples.extend(file_list[-edge_count:])
                # Random middle samples
                middle_count = count - (2 * edge_count)
                if middle_count > 0 and len(file_list) > 2 * edge_count:
                    middle = file_list[edge_count:-edge_count]
                    samples.extend(random.sample(middle, min(middle_count, len(middle))))
                return samples

            # Sample 25 from train, 25 from val (50 total)
            sample_paths.extend(stratified_sample(train_list, 25))
            sample_paths.extend(stratified_sample(val_list, 25))

            if sample_paths:
                logging.getLogger(__name__).debug(
                    f"Validating cached split (checking {len(sample_paths)} stratified sample paths with 30s timeout)..."
                )
                try:
                    from concurrent.futures import as_completed
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        # Use default argument to avoid lambda closure issues
                        futures = [executor.submit(lambda p=p: p.exists()) for p in sample_paths]
                        existence_checks = []
                        for future in as_completed(futures, timeout=30):  # 30 second timeout
                            try:
                                existence_checks.append(future.result())
                            except Exception:
                                existence_checks.append(False)
                        miss = sum(1 for exists in existence_checks if not exists)

                    if miss > 1:  # Allow 1 missing file as tolerance
                        logging.getLogger(__name__).warning(
                            f"Cached split validation failed: {miss}/{len(sample_paths)} samples missing"
                        )
                        return None
                except TimeoutError:
                    logging.getLogger(__name__).warning(
                        "Cached split validation timed out after 30s. Invalidating cache to ensure data integrity. "
                        "Consider checking filesystem health if this occurs frequently."
                    )
                    # Invalidate cache on timeout - files may have been moved/deleted
                    # Re-scanning is safer than using potentially stale cache
                    return None

            logging.getLogger(__name__).info(
                f"Using cached JSON split lists (train={len(train_list)}, val={len(val_list)})"
            )
            return train_list, val_list
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to load cached split: {e}")
    return None

def _write_cached_split(root: Path, train_list: list[Path], val_list: list[Path], seed: int = 42) -> None:
    """Write cached split files with v2.0 header. Logs warning on failure but does not raise.

    Args:
        root: Dataset root directory
        train_list: List of training file paths
        val_list: List of validation file paths
        seed: Random seed used for splitting (stored in header for validation)
    """
    train_file, val_file = _split_cache_paths(root)
    try:
        # Atomic write pattern: write to temp then rename
        train_tmp = train_file.with_suffix(".tmp")
        val_tmp = val_file.with_suffix(".tmp")

        # Build header (v2.0 format with seed)
        exclusion_hash = _compute_exclusion_hash()
        total_count = len(train_list) + len(val_list)
        header = (
            f"# SPLIT_CACHE_VERSION={_SPLIT_CACHE_VERSION}\n"
            f"# EXCLUSION_HASH={exclusion_hash}\n"
            f"# FILE_COUNT={total_count}\n"
            f"# SEED={seed}\n"
        )

        # Write with header
        train_content = header + "\n".join(str(p) for p in train_list)
        val_content = header + "\n".join(str(p) for p in val_list)

        train_tmp.write_text(train_content, encoding="utf-8")
        val_tmp.write_text(val_content, encoding="utf-8")

        train_tmp.rename(train_file)
        val_tmp.rename(val_file)

        logging.getLogger(__name__).debug(
            f"Cached split files written (train={len(train_list)}, val={len(val_list)})"
        )
    except OSError as e:
        # Disk full, permission denied, read-only filesystem
        logging.getLogger(__name__).warning(
            f"Failed to write cached split files to {train_file.parent}: {e}. "
            "Splits will be re-scanned on next run."
        )
        # Clean up partial writes
        for tmp in [train_file.with_suffix(".tmp"), val_file.with_suffix(".tmp")]:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    except (UnicodeEncodeError, ValueError) as e:
        # Path contains invalid characters or encoding issues
        logging.getLogger(__name__).warning(
            f"Failed to encode split paths: {e}. Cache disabled for this dataset."
        )


def _get_manifest_cache_path(manifest_path: Path) -> Path:
    """Return binary cache path for parsed manifest.

    Binary cache is ~2-5x faster to load than JSON and includes a checksum
    for validation.
    """
    cache_dir = _PROJ_ROOT / "logs" / "manifest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Include file size and mtime in key to auto-invalidate on changes
    try:
        stat = manifest_path.stat()
        key_str = f"{manifest_path.resolve()}_{stat.st_size}_{stat.st_mtime_ns}"
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"{key_hash}.pkl"
    except OSError:
        # If stat fails, fall back to path-only hash (cache may be stale)
        key_hash = hashlib.sha256(str(manifest_path.resolve()).encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"{key_hash}.pkl"


def _load_manifest_cached(path: Path) -> Optional[list]:
    """Try to load manifest from binary cache.

    Returns None if cache miss or invalid cache.
    """
    import pickle
    cache_path = _get_manifest_cache_path(path)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
            # Verify cache is a list (basic sanity check)
            if isinstance(cached_data, list):
                return cached_data
    except (pickle.PickleError, OSError, EOFError):
        # Cache corrupted or incompatible, will be regenerated
        pass

    return None


def _save_manifest_cache(path: Path, annotations: list) -> None:
    """Save parsed manifest to binary cache for faster future loads."""
    import pickle
    cache_path = _get_manifest_cache_path(path)
    temp_path = cache_path.with_suffix(".tmp")

    try:
        # Atomic write: write to temp then move (cross-platform safe)
        with open(temp_path, "wb") as f:
            pickle.dump(annotations, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Windows-safe atomic rename: remove existing file first if needed
        if cache_path.exists():
            cache_path.unlink()
        shutil.move(str(temp_path), str(cache_path))

        logging.getLogger(__name__).debug(
            f"Cached manifest to {cache_path} ({len(annotations)} entries)"
        )
    except (OSError, pickle.PickleError) as e:
        logging.getLogger(__name__).debug(
            f"Failed to cache manifest: {e}"
        )
        # Clean up temp file if it exists
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


class ArrowMetadataAccessor:
    """Zero-copy accessor for Arrow-backed metadata.

    Provides dict-like access to Arrow table rows without copying data.
    Used by SidecarJsonDataset to access metadata without RAM duplication
    across DataLoader workers.

    When pickled for multiprocessing, only the cache path is serialized.
    Workers re-open the memory-mapped file independently, allowing the OS
    to share the same physical memory pages across all workers.

    Memory savings: ~15 GB per worker for 5.6M images dataset.
    """

    def __init__(self, table: "pa.Table", cache_path: Path):
        """Initialize accessor.

        Args:
            table: PyArrow Table (memory-mapped)
            cache_path: Path to the Arrow IPC file (for pickling)
        """
        self._table = table
        self._cache_path = cache_path
        self._len = len(table)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a single sample.

        Returns a dict matching the legacy format:
        {"image_id": str, "tags": List[str], "rating": str, "dir": Path}
        Plus "json_stem" if available (v2.0+ cache format).
        """
        # Slice single row - Arrow handles this efficiently
        row = self._table.slice(idx, 1)
        result = {
            "image_id": row.column("image_id")[0].as_py(),
            "tags": row.column("tags")[0].as_py(),
            "rating": row.column("rating")[0].as_py(),
            "dir": Path(row.column("dir")[0].as_py()),
        }
        # Include json_stem if available (v2.0+ cache format)
        if "json_stem" in self._table.column_names:
            result["json_stem"] = row.column("json_stem")[0].as_py()
        return result

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare for pickling - only serialize the cache path."""
        return {
            "_cache_path": self._cache_path,
            "_len": self._len,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore from pickle in worker process - re-open the mmap."""
        self._cache_path = state["_cache_path"]
        self._len = state["_len"]
        # Re-open memory-mapped file in worker
        # Import here to avoid issues if pyarrow not installed
        from utils.metadata_cache import _load_arrow_cache
        import logging
        logger = logging.getLogger(__name__)
        self._table = _load_arrow_cache(self._cache_path, logger)
        if self._table is None:
            raise RuntimeError(
                f"Failed to reload Arrow metadata cache in worker process: {self._cache_path}. "
                "The cache file may be missing, corrupted, or locked. "
                "Try deleting the cache file and restarting training."
            )


class ImagePreFetcher:
    """Background thread pool for pre-fetching image files.

    Reduces I/O latency by loading the next N images in the background while
    the current batch is being processed. Uses a small thread pool and LRU
    cache to avoid memory bloat.

    Performance gains:
      - ~20-40% faster dataset iteration when cache misses are common
      - Hides disk I/O latency behind computation
      - Moves resizing/padding to background threads (new in this version)
      - Minimal memory overhead (< 100MB for typical settings)
    """

    def __init__(self, max_workers: int = 2, cache_size: int = 8, target_size: int = 512, pad_color: tuple = (114, 114, 114)):
        """Initialize pre-fetcher.

        Args:
            max_workers: Number of background threads (default 2)
            cache_size: Maximum number of pre-fetched images to cache
            target_size: Target image size for resizing/padding
            pad_color: RGB tuple for padding
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self.cache: Dict[int, Any] = {}  # idx -> (pil_img, padding_mask)
        self.cache_size = cache_size
        self.futures: Dict[int, Any] = {}  # idx -> Future
        self._lock = threading.Lock()
        self._last_idx = -1
        self.target_size = target_size
        self.pad_color = pad_color

    def _load_image(self, img_path: Path) -> Optional[Tuple[Image.Image, torch.Tensor]]:
        """Load, decode, and process image in background thread.

        Returns a fully-loaded and processed (resized/padded) image + mask.
        """
        try:
            with Image.open(img_path) as pil_img:
                pil_img.load()
                pil_img = ImageOps.exif_transpose(pil_img)

                # Handle transparency
                if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                    rgba = pil_img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, self.pad_color)
                    alpha = rgba.getchannel("A")
                    bg.paste(rgba, mask=alpha)
                    img = bg
                else:
                    img = pil_img.convert("RGB")
                
                # Perform resizing and padding in background thread
                return process_image_cpu(img, self.target_size, self.pad_color)
        except Exception:
            # Silent failure - will be retried in main thread
            return None

    def prefetch(self, idx: int, img_path: Path) -> None:
        """Start pre-fetching an image in the background.

        Args:
            idx: Dataset index for this image
            img_path: Path to image file
        """
        with self._lock:
            # Only prefetch if not already cached or in-flight
            if idx not in self.cache and idx not in self.futures:
                future = self.executor.submit(self._load_image, img_path)
                self.futures[idx] = (future, img_path)

    def get(self, idx: int, wait_timeout: Optional[float] = 0.05) -> Optional[Tuple[Image.Image, torch.Tensor]]:
        """Get pre-fetched image, optionally waiting for completion.

        Args:
            idx: Dataset index to retrieve
            wait_timeout: Max seconds to wait for prefetch (default 0.05s = 50ms).
                         Set to 0 for non-blocking, None for unlimited wait.

        Returns:
            (PIL Image, padding_mask) if pre-fetched, None otherwise
        """
        future_to_wait = None

        with self._lock:
            # Check cache first
            if idx in self.cache:
                return self.cache.pop(idx)

            # Check if future exists
            if idx in self.futures:
                future, img_path = self.futures[idx]

                if future.done():
                    # Future completed - pop and return
                    self.futures.pop(idx)
                    try:
                        result = future.result(timeout=0)
                        return result
                    except Exception:
                        return None

                # Future not done - optionally wait outside lock to avoid blocking
                # wait_timeout: None=unlimited, >0=wait with timeout, 0=non-blocking
                if wait_timeout is None or wait_timeout > 0:
                    # Pop from futures while holding lock (safe), wait outside lock
                    self.futures.pop(idx)
                    future_to_wait = future

        # Wait for future OUTSIDE of lock (no race condition since we already popped it)
        if future_to_wait is not None:
            try:
                # None means unlimited wait, positive means timeout in seconds
                return future_to_wait.result(timeout=wait_timeout)
            except Exception:
                return None

        return None

    def trigger_lookahead(self, current_idx: int, dataset_len: int,
                         get_path_func, lookahead: int = 4) -> None:
        """Trigger pre-fetching for upcoming indices.

        Args:
            current_idx: Current dataset index being accessed
            dataset_len: Total dataset length
            get_path_func: Function to get image path for an index (idx -> Path)
            lookahead: Number of indices to pre-fetch ahead (default 4)
        """
        # Detect sequential access pattern (thread-safe read of _last_idx)
        with self._lock:
            last_idx = self._last_idx
            self._last_idx = current_idx

        if current_idx == last_idx + 1 or last_idx < 0:
            # Pre-fetch next N images
            for offset in range(1, lookahead + 1):
                next_idx = current_idx + offset
                if next_idx < dataset_len:
                    try:
                        img_path = get_path_func(next_idx)
                        self.prefetch(next_idx, img_path)
                    except Exception:
                        pass  # Invalid path, skip

        # Cleanup: remove stale futures and enforce cache size
        with self._lock:
            # Check for completed futures
            # We must process ALL completed futures to prevent memory leaks in self.futures
            # If cache is full, we still need to remove them from self.futures (discarding the result)
            idxs_to_remove = []
            
            for idx, (future, _) in list(self.futures.items()):  # List copy for safe iteration
                if future.done():
                    idxs_to_remove.append(idx)
                    
            for idx in idxs_to_remove:
                future, img_path = self.futures.pop(idx)
                # Only add to cache if we have space
                if len(self.cache) < self.cache_size:
                    try:
                        result = future.result(timeout=0)
                        if result is not None:
                            self.cache[idx] = result
                    except Exception:
                        pass
                # else: implicit discard (future removed from futures, result ignored)

            # Evict old cache entries if we exceed size
            while len(self.cache) > self.cache_size:
                # Remove oldest (arbitrary key)
                self.cache.pop(next(iter(self.cache)))

    def shutdown(self) -> None:
        """Shutdown background threads and cleanup resources."""
        self.executor.shutdown(wait=True)
        # Close any cached PIL Images to release file handles
        for item in self.cache.values():
            if isinstance(item, tuple):
                 img = item[0]
            else:
                 img = item
            if hasattr(img, 'close'):
                try:
                    img.close()
                except Exception:
                    pass
        self.cache.clear()
        self.futures.clear()


class WorkerInitializer:
    """Picklable worker initialization callable for DataLoader.

    Handles logging queue setup and shared vocabulary loading in worker processes.
    Unlike closures, class instances are picklable by default.
    """

    def __init__(self, log_queue=None, shared_vocab_info=None):
        """
        Args:
            log_queue: Queue for logging (optional)
            shared_vocab_info: Tuple of (shm_name, vocab_size) for shared vocabulary (optional)
        """
        self.log_queue = log_queue
        self.shared_vocab_info = shared_vocab_info

    def __call__(self, worker_id: int):
        """Worker initialization function called by DataLoader.

        Args:
            worker_id: Worker process ID
        """
        # Setup logging queue handler
        if self.log_queue is not None:
            logger = logging.getLogger()
            # Ensure a single QueueHandler per worker
            for h in list(logger.handlers):
                try:
                    from logging.handlers import QueueHandler  # local import to avoid import-time dependency
                    if isinstance(h, QueueHandler):
                        logger.removeHandler(h)
                except Exception:
                    # Fallback: check class name to avoid hard import
                    if getattr(h, "__class__", None) and h.__class__.__name__ == "QueueHandler":
                        logger.removeHandler(h)
            try:
                from logging.handlers import QueueHandler
                logger.addHandler(QueueHandler(self.log_queue))
                # Set worker log level to CRITICAL to minimize queue traffic from workers
                # WARNING-level logs still have overhead (queue serialization, main process dequeue)
                # With 8+ workers, even infrequent warnings add measurable latency (~1-2ms/batch)
                logger.setLevel(logging.CRITICAL)
            except Exception:
                pass

        # Load shared vocabulary if available
        if self.shared_vocab_info is not None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            if worker_info is not None:
                dataset = worker_info.dataset
                shm_name, vocab_size = self.shared_vocab_info

                # Check if dataset has vocab attribute and needs loading
                if hasattr(dataset, 'vocab') and hasattr(dataset, '_shared_vocab_loaded'):
                    if not dataset._shared_vocab_loaded:
                        try:
                            from shared_vocabulary import SharedVocabularyManager, populate_vocab_from_shared
                            # Load vocabulary from shared memory
                            vocab_data = SharedVocabularyManager.load_from_shared(shm_name, vocab_size)
                            populate_vocab_from_shared(dataset.vocab, vocab_data)
                            dataset._shared_vocab_loaded = True
                            logging.getLogger(__name__).debug(
                                f"Worker {worker_id}: Loaded vocabulary from shared memory ({vocab_size / 1024:.1f} KB)"
                            )
                        except Exception as e:
                            logging.getLogger(__name__).warning(
                                f"Worker {worker_id}: Failed to load shared vocabulary: {e}"
                            )


class DatasetLoader(Dataset):
    def __init__(
        self,
        annotations_path,
        image_dir,
        dataset_root: Optional[str] = None,
        transform=None,
        joint_transforms=None,  # NEW: torchvision v2 transforms applied to (image, mask) together
        max_retries=2,
        num_classes=None,
        # Image pipeline params
        image_size: int = 512,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        preload_files: int = 0,
        # Background validator control
        enable_background_validator: Optional[bool] = None,
        # Dtype configuration
        tag_vector_dtype: str = "bfloat16",
    ):
        """
        Dataset loader for images and JSON metadata.
        Note: Despite legacy naming, this does NOT handle HDF5 files.
        """
        self.annotations = self._load_annotations(annotations_path)
        self.image_dir = image_dir
        self.transform = transform
        self.joint_transforms = joint_transforms
        self.max_retries = max_retries
        # Validate num_classes to prevent dimension mismatches in tag vectors
        if num_classes is None:
            logging.warning(
                "DatasetLoader: num_classes not provided. Tag vectors may have incorrect dimensions. "
                "Pass num_classes=len(vocab.tag_to_index) for consistency."
            )
        self.num_classes = num_classes
        self.retry_counts = {}
        self.failed_samples = set()
        self._sample_error_log_count = 0  # Rate-limit error logs
        self.logger = logging.getLogger(__name__)
        # Track error distribution per tag to detect bias
        from collections import defaultdict
        self.error_stats = defaultdict(lambda: defaultdict(int))
        self._error_warn_counts = defaultdict(int)  # Rate limit warnings per tag
        # For manifest mode, allow symlink targets to resolve within this dataset root
        self.dataset_root = dataset_root

        # Exclusion manager for bad/corrupted images - supports live persistence
        # and periodic reload for multi-worker synchronization
        exclusion_base = Path(dataset_root) if dataset_root else Path(image_dir).parent
        exclusion_path = exclusion_base / 'cache_exclusions.txt'
        self._exclusion_manager = ExclusionManager(
            exclusion_path,
            reload_interval_seconds=_EXCLUSION_RELOAD_INTERVAL
        )
        self.excluded_image_ids = self._exclusion_manager.load()
        if self.excluded_image_ids:
            self.logger.info(f"Loaded {len(self.excluded_image_ids)} excluded image IDs from {exclusion_path}")

        # Image pipeline settings
        self.image_size = int(image_size)
        self.pad_color: Tuple[int, int, int] = (
            int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
        ) if isinstance(pad_color, (list, tuple)) else (114, 114, 114)
        self.normalize_mean: Tuple[float, float, float] = tuple(normalize_mean)
        self.normalize_std: Tuple[float, float, float] = tuple(normalize_std)

        # CPU BF16 cache pipeline (disabled for manifest mode DatasetLoader)
        self._cpu_bf16_cache = False

        # Tag vector dtype
        self._tag_vector_dtype = _canon_dtype(str(tag_vector_dtype).lower())

        # Properly initialise background validator (opt-out via env or param)
        if enable_background_validator is None:
            enable_background_validator = os.getenv("DATASET_BACKGROUND_VALIDATOR", "1") != "0"

        # Check if we're in a worker process - if so, disable validator to avoid fork issues
        worker_info = torch.utils.data.get_worker_info()
        in_worker = worker_info is not None

        # Only create validator in main process
        self.validator = None
        if enable_background_validator and not in_worker:
            self.validator = BackgroundValidator(self)
            self.validator.start()
        elif enable_background_validator and in_worker:
            self.logger.debug(
                "Disabled BackgroundValidator in DataLoader worker process"
            )

        # Epoch tracking for future flip support and consistency with SidecarJsonDataset
        # Default to 0 for first epoch; updated by set_epoch() in training loop
        self._current_epoch = 0

        self._preload_n = int(preload_files or 0)

        # --- Image pre-fetching (per-worker; created lazily) ---
        self._prefetcher: Optional[ImagePreFetcher] = None
        self._enable_prefetch = os.getenv("DATASET_PREFETCH", "1") != "0"

        # --- Shared vocabulary flag (for worker_init_fn) ---
        self._shared_vocab_loaded = False

        # --- Pre-created transforms for performance (avoid recreating per sample) ---
        # Use v2 API to avoid deprecation warning (ToTensor is deprecated)
        if T is not None:
            self._to_tensor_v2 = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
            self._to_tensor = self._to_tensor_v2  # Use v2 for v1 fallback too
        elif transforms is not None:
            self._to_tensor_v2 = None
            self._to_tensor = transforms.ToTensor()  # Legacy fallback
        else:
            self._to_tensor_v2 = None
            self._to_tensor = None

    # ---------- Pre-fetcher ----------
    def _ensure_prefetcher(self):
        """Create image pre-fetcher lazily (per-worker)."""
        if not self._enable_prefetch or self._prefetcher is not None:
            return
        # Configurable prefetch settings (increased defaults to hide I/O latency)
        # DATASET_PREFETCH_SIZE: Number of images to prefetch (default 32, was 8)
        # DATASET_PREFETCH_WORKERS: Number of background threads (default 2)
        # Reduced from 4 to avoid thread oversubscription with num_workers=6 (would be 24+6=30 threads)
        # Memory overhead: ~3MB per cached image, so 32 images = ~96MB per worker
        cache_size = int(os.getenv("DATASET_PREFETCH_SIZE", "32"))
        max_workers = int(os.getenv("DATASET_PREFETCH_WORKERS", "2"))
        self._prefetcher = ImagePreFetcher(
            max_workers=max_workers, 
            cache_size=cache_size,
            target_size=self.image_size,
            pad_color=self.pad_color
        )

    # ---------- Pickling support for multiprocessing ----------
    def __getstate__(self):
        """Prepare for pickling - exclude unpicklable objects."""
        state = self.__dict__.copy()
        # Remove unpicklable objects before sending to worker
        state['validator'] = None           # BackgroundValidator thread
        state['_prefetcher'] = None        # ImagePreFetcher thread pool
        return state

    def __setstate__(self, state):
        """Restore from pickle in worker process."""
        self.__dict__.update(state)
        # These will be lazily recreated when needed:
        # - _prefetcher via _ensure_prefetcher()
        # - validator stays None in workers

    def _get_image_path_for_idx(self, idx: int) -> Path:
        """Get image path for a given index (for prefetching).

        Returns:
            Path to image file

        Raises:
            Exception on any error (caught by prefetcher)
        """
        annotation = self.annotations[idx]
        raw_image_id = sanitize_identifier(str(annotation['image_id']))
        return validate_image_path(
            Path(self.image_dir),
            raw_image_id,
            allowed_external_roots=([Path(self.dataset_root)] if self.dataset_root else None),
        )

    def _encode_labels(self, annotation: Dict[str, Any]) -> torch.Tensor:
        """Encode tag labels from annotation to multi-hot vector.

        Args:
            annotation: Annotation dict with 'labels' field

        Returns:
            Multi-hot tensor of shape (num_classes,)
        """
        tag_indices = annotation.get("labels") or []

        # Validate indices and create multi-hot vector
        if (
            isinstance(tag_indices, list)
            and len(tag_indices) > 0
            and isinstance(tag_indices[0], (int, float))
            and self.num_classes
        ):
            # Filter invalid indices
            valid_indices = [
                int(i) for i in tag_indices
                if 0 <= int(i) < self.num_classes
            ]

            tag_vec = torch.zeros(self.num_classes, dtype=self._tag_vector_dtype)
            if valid_indices:
                tag_vec.scatter_(
                    0,
                    torch.tensor(valid_indices, dtype=torch.long),
                    1.0,
                )
        else:
            # No valid labels - return zero vector
            tag_vec = torch.zeros(self.num_classes or 1, dtype=self._tag_vector_dtype)

        return tag_vec

    def _build_sample_dict(
        self,
        image: torch.Tensor,
        padding_mask: torch.Tensor,
        annotation: Dict[str, Any],
        image_id: str,
        cached: bool = False,
    ) -> Dict[str, Any]:
        """Build the sample dictionary returned by __getitem__.

        Args:
            image: Preprocessed image tensor (C, H, W)
            padding_mask: Boolean padding mask (H, W)
            annotation: Annotation dict with labels and rating
            image_id: Image identifier
            cached: Whether data came from cache

        Returns:
            Sample dict for training
        """
        tag_vec = self._encode_labels(annotation)
        rating_idx = _map_rating(annotation.get("rating", "unknown"))

        # Ensure tensors are contiguous before returning for efficient pin_memory
        # Non-contiguous tensors force implicit copies during DataLoader collation/pinning
        if not image.is_contiguous():
            image = image.contiguous()
        if not padding_mask.is_contiguous():
            padding_mask = padding_mask.contiguous()

        return {
            "images": image,
            "padding_mask": padding_mask.to(torch.bool),
            "tag_labels": tag_vec,
            "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
            "image_id": image_id,
            "cached": cached,
            "error": False,
            "error_reason": "",
        }

    def preload_first_n(self, n: int):
        n = int(max(0, n))
        if n == 0 or len(self) == 0:
            return
        for i in range(min(n, len(self))):
            _ = self[i]

    def _load_annotations(self, path):
        """Load annotation JSON file with validation and binary caching.

        Optimizations:
          - Binary cache (pickle) for 2-5x faster subsequent loads
          - Automatic cache invalidation on file changes (mtime + size)
          - Atomic writes to prevent corruption

        Args:
            path: Path to annotations JSON file

        Returns:
            List of annotation dictionaries

        Raises:
            FileNotFoundError: If annotation file doesn't exist
            ValueError: If JSON is malformed or has wrong structure
            RuntimeError: For other I/O errors
        """
        path_obj = Path(path)

        # Check file exists first for clearer error message
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {path}\n"
                f"Please check the path and ensure the file exists."
            )

        # Check file is readable
        if not path_obj.is_file():
            raise ValueError(
                f"Annotation path is not a file: {path}\n"
                f"Expected a JSON file, got: {path_obj}"
            )

        # Try to load from binary cache first (2-5x faster than JSON)
        annotations = _load_manifest_cached(path_obj)
        if annotations is not None:
            self.logger.info(
                f"Loaded {len(annotations)} annotations from cache (fast path)"
            )
            # Skip validation for cached data (already validated when cached)
            return annotations

        # Cache miss: load from JSON (use orjson if available for 3-5x speedup)
        try:
            if HAS_ORJSON:
                annotations = orjson.loads(path_obj.read_bytes())
            else:
                with open(path_obj, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
        except JSON_DECODE_ERRORS as e:
            raise ValueError(
                f"Failed to parse annotation JSON file: {path}\n"
                f"JSON syntax error: {e}\n"
                f"Please validate the JSON file using a JSON linter."
            ) from e
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Failed to decode annotation file: {path}\n"
                f"File encoding error: {e}\n"
                f"Expected UTF-8 encoded JSON file. Try opening in a text editor "
                f"and saving as UTF-8."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Failed to read annotation file: {path}\n"
                f"I/O error: {e}\n"
                f"Check file permissions and disk status."
            ) from e

        # Validate structure
        if not isinstance(annotations, list):
            raise ValueError(
                f"Invalid annotation file structure: {path}\n"
                f"Expected a JSON list, got: {type(annotations).__name__}\n"
                f"Annotation files should contain a list of image metadata objects."
            )

        if len(annotations) == 0:
            self.logger.warning(
                f"Annotation file is empty: {path}\n"
                f"No samples will be loaded from this dataset."
            )

        # Basic validation of first entry
        if len(annotations) > 0:
            sample = annotations[0]
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Invalid annotation entry in: {path}\n"
                    f"First entry is {type(sample).__name__}, expected dict\n"
                    f"Each annotation should be a JSON object with image_id, labels, etc."
                )
            if "image_id" not in sample:
                raise ValueError(
                    f"Missing required 'image_id' field in: {path}\n"
                    f"First annotation entry: {sample}\n"
                    f"Each annotation must have an 'image_id' field."
                )

        self.logger.info(f"Loaded {len(annotations)} annotations from {path}")

        # Save to binary cache for faster future loads (async to avoid blocking)
        # Note: cache write failures are logged but don't fail the operation
        _save_manifest_cache(path_obj, annotations)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for potential future flip support.

        Currently DatasetLoader (manifest mode) does not support flipping,
        but this method is provided for API consistency with SidecarJsonDataset
        and future extensibility.

        Args:
            epoch: Current training epoch (0-indexed)
        """
        self._current_epoch = int(epoch)
        self.logger.debug(f"Dataset epoch set to {self._current_epoch}")

    def __getitem__(self, idx):
        # Periodically reload exclusions to pick up failures from other workers
        if self._exclusion_manager:
            if self._exclusion_manager.reload_if_stale():
                new_exclusions = self._exclusion_manager.get_excluded_ids()
                if len(new_exclusions) > len(self.excluded_image_ids):
                    self.excluded_image_ids = new_exclusions

        # HL002 Fix: Return error sample immediately on failure, don't bias distribution
        if idx in self.failed_samples:
            return self._create_error_sample(idx, "Previously failed sample")

        # Check if this sample was excluded by another worker (cross-worker sync)
        if idx < len(self.annotations):
            item_image_id = sanitize_identifier(str(self.annotations[idx].get('image_id', '')))
            if item_image_id and item_image_id in self.excluded_image_ids:
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)
                return self._create_error_sample(idx, f"Excluded: {item_image_id}")

        # Track retries with memory bounds to prevent unbounded growth
        if idx not in self.retry_counts:
            # Evict oldest entries if at capacity (simple FIFO-like eviction)
            if len(self.retry_counts) >= _MAX_RETRY_COUNTS:
                # Remove ~10% of entries to amortize eviction cost
                keys_to_remove = list(self.retry_counts.keys())[:_MAX_RETRY_COUNTS // 10]
                for k in keys_to_remove:
                    del self.retry_counts[k]
            self.retry_counts[idx] = 0

        try:
            annotation = self.annotations[idx]
            # Enforce allowlist and strip any sneaky path components
            raw_image_id = sanitize_identifier(str(annotation['image_id']))

            # --- Load + transform (confined path) ---
            # Use the sanitized image identifier we derived above.
            # Allow symlink targets to live under the dataset root (manifest symlinks → shard files)
            img_path = validate_image_path(
                Path(self.image_dir),
                raw_image_id,
                allowed_external_roots=([Path(self.dataset_root)] if self.dataset_root else None),
            )

            # Try to use pre-fetched image first (reduces I/O latency by ~20-40%)
            self._ensure_prefetcher()
            prefetch_result = None
            if self._prefetcher is not None:
                prefetch_result = self._prefetcher.get(idx)
                # Trigger lookahead for next images
                self._prefetcher.trigger_lookahead(
                    idx, len(self), self._get_image_path_for_idx
                )

            # If not pre-fetched, load synchronously
            if prefetch_result is None:
                # Fully decode while file is open; fix EXIF rotations.
                with Image.open(img_path) as pil_img:
                    pil_img.load()
                    pil_img = ImageOps.exif_transpose(pil_img)

                    if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                        rgba = pil_img.convert("RGBA")
                        bg = Image.new("RGB", rgba.size, self.pad_color)
                        alpha = rgba.getchannel("A")
                        bg.paste(rgba, mask=alpha)
                        img = bg
                    else:
                        img = pil_img.convert("RGB")
                
                # Process on CPU (resize/pad)
                canvas, pmask = process_image_cpu(img, self.image_size, self.pad_color)
            else:
                canvas, pmask = prefetch_result

            # If provided, run joint v2 transforms to keep image & mask aligned
            if self.joint_transforms is not None and T is not None and tv_tensors is not None:
                img_tv = tv_tensors.Image(canvas)
                mask_tv = tv_tensors.Mask(pmask.to(torch.uint8))  # 1=PAD, 0=valid
                # v2 ops automatically use NEAREST for Mask; geometry stays in sync
                img_tv, mask_tv = self.joint_transforms(img_tv, mask_tv)
                # Pre-norm 0..1 tensor for L1; then normalize for model
                img_01 = self._to_tensor_v2(img_tv)  # 0..1 float
                if self._cpu_bf16_cache:
                    img_01 = img_01.to(torch.bfloat16)
                t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)
                pmask = mask_tv.to(torch.bool)
            else:
                # Fallback: color-only transforms ok; any geometry here would desync pmask
                if self.transform:
                    try:
                        transformed = self.transform(canvas)
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        # Ensure we can derive 0..1 image for L1 regardless of transform type
                        img_01 = transformed if isinstance(transformed, torch.Tensor) else self._to_tensor(transformed)
                        # Keep dtype as-is; bf16 preferred when configured
                        if self._cpu_bf16_cache:
                            img_01 = img_01.to(torch.bfloat16)
                        t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)
                    except Exception as e:
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Transform failed, using fallback: {e}")
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img_01 = self._to_tensor(canvas)
                        if self._cpu_bf16_cache:
                            img_01 = img_01.to(torch.bfloat16)
                        t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)
                else:
                    if self._to_tensor is None:
                        raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                    img_01 = self._to_tensor(canvas)
                    if self._cpu_bf16_cache:
                        img_01 = img_01.to(torch.bfloat16)
                    t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)

            # Reset retry count on success
            self.retry_counts[idx] = 0

            # Build sample using helper to avoid duplication
            return self._build_sample_dict(
                t, pmask, annotation, raw_image_id, cached=False
            )

        except Exception as e:
            self.retry_counts[idx] += 1
            self._sample_error_log_count += 1
            # Rate-limit warning logs: log first, then every 100th
            if self._sample_error_log_count == 1 or self._sample_error_log_count % 100 == 0:
                self.logger.warning(f"Failed to load sample {idx}: {e} (total errors: {self._sample_error_log_count})")

            # Track error distribution to detect bias
            error_type = 'load_failed' if 'load' in str(e).lower() else 'decode_failed'
            self._track_error_distribution(idx, error_type)

            if self.retry_counts[idx] >= self.max_retries:
                # Add to failed set with memory bounds
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)

                # Persist failed sample to exclusion file immediately
                try:
                    failed_image_id = sanitize_identifier(str(self.annotations[idx].get('image_id', '')))
                    if failed_image_id and self._exclusion_manager:
                        was_new = self._exclusion_manager.add_exclusion(failed_image_id, immediate=True)
                        if was_new:
                            self.excluded_image_ids.add(failed_image_id)
                            self.logger.info(
                                f"Persisted exclusion for {failed_image_id} (sample {idx}) - "
                                f"will be skipped in future runs"
                            )
                except Exception as persist_err:
                    self.logger.warning(f"Could not persist exclusion for sample {idx}: {persist_err}")

                # Always log when sample permanently fails (rate-limited by max_retries)
                self.logger.error(f"Sample {idx} exceeded max retries, marking as failed")
                return self._create_error_sample(idx, str(e))

            # Return error sample instead of silently advancing to next index
            return self._create_error_sample(idx, f"Temporary failure: {e}")

    def _track_error_distribution(self, idx: int, error_type: str):
        """Track error rates per tag to detect distribution bias.

        Args:
            idx: Sample index that failed
            error_type: Type of error (e.g., 'load_failed', 'decode_failed')
        """
        # Enforce memory bounds on error tracking structures
        if len(self.error_stats) >= _MAX_ERROR_STATS_TAGS:
            # Stop tracking new tags once limit reached to prevent memory bloat
            # Existing tags continue to be tracked
            pass  # Will skip adding new tags below

        # Bounds check with try-except for safety in case of concurrent access
        try:
            if idx < 0 or idx >= len(self.annotations):
                return
            annotation = self.annotations[idx]
        except (IndexError, TypeError):
            return

        tag_indices = annotation.get("labels") or []

        # Track errors for each tag in this sample
        for tag_idx in tag_indices:
            if isinstance(tag_idx, (int, float)) and self.num_classes:
                tag_idx = int(tag_idx)
                if 0 <= tag_idx < self.num_classes:
                    # Only track if tag already tracked or we have room for new tags
                    if tag_idx not in self.error_stats and len(self.error_stats) >= _MAX_ERROR_STATS_TAGS:
                        continue  # Skip new tags when at capacity

                    self.error_stats[tag_idx][error_type] += 1
                    self.error_stats[tag_idx]['total'] += 1

                    # Log warning if error rate exceeds threshold (rate-limited)
                    total_errors = self.error_stats[tag_idx]['total']
                    if total_errors > 50 and total_errors % 25 == 0:  # Check every 25 errors after 50
                        error_rate = self.error_stats[tag_idx][error_type] / total_errors
                        if error_rate > 0.1:  # >10% error rate
                            # Rate limit: only warn once per 100 errors for each tag
                            if self._error_warn_counts[tag_idx] < total_errors // 100:
                                self._error_warn_counts[tag_idx] += 1
                                self.logger.warning(
                                    f"Tag index {tag_idx} has high error rate: "
                                    f"{error_rate:.1%} {error_type} errors "
                                    f"({self.error_stats[tag_idx][error_type]}/{total_errors} samples). "
                                    f"This may bias training distribution."
                                )

    def _create_error_sample(self, idx, reason):
        """Create a clearly marked error sample"""
        # Default to a common square size when transform is unknown
        sz = int(getattr(self, "image_size", 224) or 224)
        # Ensure num_classes is valid to prevent shape mismatches during batching
        if not self.num_classes or self.num_classes <= 0:
            raise ValueError(
                f"Cannot create error sample: num_classes={self.num_classes} is invalid. "
                "Pass num_classes=len(vocab.tag_to_index) when creating DatasetLoader."
            )
        return {
            "images": torch.zeros((3, sz, sz)),  # Placeholder tensor
            "padding_mask": torch.ones((sz, sz), dtype=torch.bool),
            "tag_labels": torch.zeros(self.num_classes, dtype=self._tag_vector_dtype),
            "rating_labels": torch.tensor(4, dtype=torch.long),  # unknown
            "image_id": f"error_{idx}",
            "cached": False,
            "flip_applied": False,
            "flip_mode": "none",
            "has_tag_mismatch": False,
            "error": True,
            "error_reason": reason,
        }

    def get_failure_statistics(self):
        """Return statistics about failed samples for logging."""
        # Calculate per-tag error rates
        tag_error_summary = {}
        for tag_idx, stats in self.error_stats.items():
            total = stats['total']
            if total > 0:
                tag_error_summary[tag_idx] = {
                    'total_errors': total,
                    'load_failed': stats.get('load_failed', 0),
                    'decode_failed': stats.get('decode_failed', 0),
                    'error_rate': total / len(self.annotations) if len(self.annotations) > 0 else 0,
                }

        return {
            "total_failed": len(self.failed_samples),
            "failed_indices": list(self.failed_samples),
            "retry_counts": self.retry_counts,
            "tag_error_distribution": tag_error_summary,
        }

    def close(self):
        """Release resources."""
        # Cleanup validator
        if hasattr(self, "validator") and self.validator is not None:
            try:
                self.validator.close()
                # Give thread time to finish
                if hasattr(self.validator, "join"):
                    self.validator.join(timeout=2.0)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    f"Error stopping validator: {e}"
                )
            finally:
                self.validator = None

    def __del__(self):
        """Fallback cleanup when object is garbage collected."""
        try:
            self.close()
        except Exception:
            pass  # Silently ignore errors in __del__

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False


_active_validators = weakref.WeakSet()


class BackgroundValidator(Thread):
    """Background validation thread with explicit cleanup and registry tracking."""

    def __init__(self, dataset_loader):
        # Use daemon=True so thread doesn't block process exit
        # Validation is non-critical and can be safely interrupted
        super().__init__(daemon=True)
        self.dataset_loader = dataset_loader
        self.validation_queue = queue.Queue(maxsize=1000)
        self.running = True
        self._stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        _active_validators.add(self)

    def run(self):
        """Background validation loop"""
        self.logger.debug("BackgroundValidator thread started")
        while self.running and not self._stop_event.is_set():
            try:
                # Use timeout to check stop_event periodically
                try:
                    item_idx = self.validation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item_idx is None:  # Sentinel for shutdown
                    break

                self.validate_item(item_idx)
                self.validation_queue.task_done()

            except Exception as e:
                self.logger.error(f"Validation error: {e}")

        self.logger.debug("BackgroundValidator thread stopping")

    def validate_item(self, idx):
        """Perform actual validation of dataset items"""
        try:
            annotation = self.dataset_loader.annotations[idx]

            # Confine and locate image file safely
            try:
                image_id = sanitize_identifier(str(annotation["image_id"]))
                image_path = validate_image_path(
                    Path(self.dataset_loader.image_dir),
                    image_id,
                    allowed_external_roots=([Path(self.dataset_loader.dataset_root)] if getattr(self.dataset_loader, 'dataset_root', None) else None),
                )
            except Exception as e:
                logging.warning(f"Invalid image_id for item {idx}: {e}")
                return False

            # Validate image can be opened
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ["RGB", "L"]:
                        logging.warning(f"Unexpected image mode {img.mode} for {image_path}")
            except Exception as e:
                logging.warning(f"Cannot open image {image_path}: {e}")
                return False

            # Validate labels are within expected range
            if "labels" in annotation and self.dataset_loader.num_classes is not None:
                labels = annotation["labels"]
                try:
                    if not all(0 <= int(label) < int(self.dataset_loader.num_classes) for label in labels):
                        logging.warning(f"Invalid labels for item {idx}: {labels}")
                        return False
                except Exception:
                    return False

            return True

        except Exception as e:
            logging.error(f"Validation failed for item {idx}: {e}")
            return False

    def stop(self, timeout=5.0):
        """Stop the validation thread gracefully."""
        if not self.running:
            return  # Already stopped

        self.logger.debug("Stopping BackgroundValidator...")
        self.running = False
        self._stop_event.set()

        # Send sentinel to unblock queue.get()
        try:
            self.validation_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for thread to finish
        if self.is_alive():
            self.join(timeout=timeout)
            if self.is_alive():
                self.logger.warning(
                    f"BackgroundValidator did not stop within {timeout}s timeout"
                )

        # Clean up queue
        try:
            while not self.validation_queue.empty():
                self.validation_queue.get_nowait()
        except queue.Empty:
            pass

    def close(self):
        """Explicit cleanup method for resource management."""
        if not self.running:
            return  # Already closed

        self.logger.debug("Closing BackgroundValidator...")
        self.stop(timeout=10.0)  # Longer timeout for explicit cleanup

        # Remove from registry
        _active_validators.discard(self)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - guaranteed cleanup."""
        self.close()
        return False

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            if self.running:
                self.stop(timeout=1.0)
                _active_validators.discard(self)
        except Exception:
            pass


@atexit.register
def _cleanup_all_validators():
    """Emergency cleanup of any remaining validators."""
    for validator in list(_active_validators):
        try:
            validator.close()
        except Exception:
            pass


class AugmentationStats:
    """Placeholder class for augmentation statistics."""
    pass


def validate_dataset(*args, **kwargs):
    """Placeholder dataset validation function."""
    return {}


class SidecarJsonDataset(Dataset):
    """Dataset that reads per-image JSON sidecars in the same folder as images.

    Each JSON is expected to contain at least:
      - filename: image file name (e.g., "12345.jpg")
      - tags: space-delimited string or list of tags
      - rating: optional rating string or int (safe/general/questionable/explicit/unknown)

    Supports sidecar tensor cache: preprocessed images are stored as .safetensor files
    alongside original images for fast loading on subsequent runs.
    """

    def __init__(
        self,
        root_dir: Path,
        json_files: List[Path],
        vocab: TagVocabulary,
        transform=None,
        joint_transforms=None,  # NEW
        max_retries: int = 2,
        # Image pipeline params
        image_size: int = 512,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        # Sidecar cache configuration (replaces L2 LMDB cache)
        sidecar_cache_enabled: bool = True,
        sidecar_extension: str = ".safetensor",
        sidecar_storage_dtype: str = "bfloat16",
        cpu_bf16_cache_pipeline: Optional[bool] = None,
        # --- Orientation / flipping ---
        random_flip_prob: float = 0.0,
        orientation_handler: Optional["OrientationHandler"] = None,
        flip_overrides_path: Optional[str] = None,   # JSON with {"force_flip":[ids], "never_flip":[ids]} (also accepts {"flip":[...]} or a bare list)
        respect_flip_list: bool = True,
        stats_queue: Optional[mp.Queue] = None,
        # Dtype configuration
        tag_vector_dtype: str = "bfloat16",
        # Metadata cache configuration
        metadata_cache_enabled: bool = True,
        metadata_cache_workers: int = 16,
        force_rebuild_metadata_cache: bool = False,
        metadata_cache_staleness_check_samples: int = 100,
        prebuilt_arrow_table: Optional[Any] = None,  # Pre-loaded Arrow table to avoid rebuild
    ):
        self.root = Path(root_dir)
        self.json_files = list(json_files)
        self.vocab = vocab
        self.transform = transform
        self.joint_transforms = joint_transforms
        self.max_retries = max_retries
        self.retry_counts: Dict[int, int] = {}
        self.failed_samples = set()
        self._sample_error_log_count = 0  # Rate-limit error logs
        self.logger = logging.getLogger(__name__)

        # Exclusion manager for bad/corrupted images - supports live persistence
        # and periodic reload for multi-worker synchronization
        exclusion_path = self.root / 'cache_exclusions.txt'
        self._exclusion_manager = ExclusionManager(
            exclusion_path,
            reload_interval_seconds=_EXCLUSION_RELOAD_INTERVAL
        )
        self.excluded_image_ids = self._exclusion_manager.load()
        if self.excluded_image_ids:
            self.logger.info(f"Loaded {len(self.excluded_image_ids)} excluded image IDs from {exclusion_path}")

        # Image pipeline settings
        self.image_size = int(image_size)
        self.pad_color: Tuple[int, int, int] = (
            int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
        ) if isinstance(pad_color, (list, tuple)) else (114, 114, 114)
        self.normalize_mean: Tuple[float, float, float] = tuple(normalize_mean)
        self.normalize_std: Tuple[float, float, float] = tuple(normalize_std)

        # Sidecar cache configuration
        self._sidecar_enabled = bool(sidecar_cache_enabled)
        self._sidecar_extension = str(sidecar_extension) if sidecar_extension.startswith(".") else f".{sidecar_extension}"
        self._sidecar_dtype_str = str(sidecar_storage_dtype).lower()
        self._sidecar_dtype = _canon_dtype(self._sidecar_dtype_str)

        # Tag vector dtype
        self._tag_vector_dtype = _canon_dtype(str(tag_vector_dtype).lower())

        # Check CPU BF16 support before enabling
        requested_cpu_bf16 = bool(cpu_bf16_cache_pipeline) if cpu_bf16_cache_pipeline is not None else (self._sidecar_dtype is torch.bfloat16)

        if requested_cpu_bf16:
            if _is_cpu_bf16_supported():
                self._cpu_bf16_cache = True
                self.logger.debug("CPU BF16 cache pipeline enabled")
            else:
                self._cpu_bf16_cache = False
                # Also fall back storage dtype to float32 to ensure cache consistency
                # This prevents issues where bf16 tensors are stored but operations fail
                if self._sidecar_dtype is torch.bfloat16:
                    self._sidecar_dtype = torch.float32
                    self._sidecar_dtype_str = "float32"
                self.logger.warning(
                    "CPU does not support bfloat16 operations. "
                    "Falling back to float32 for both cache pipeline and storage. "
                    "IMPACT: Existing bfloat16 cache files will be automatically re-processed during training "
                    "because the storage_dtype change affects the config hash (from 'bfloat16' to 'float32'). "
                    "No manual cache deletion is needed - stale entries are detected and replaced. "
                    "To suppress this warning, set sidecar_storage_dtype='float32' in config."
                )
        else:
            self._cpu_bf16_cache = False
            # Force sidecar dtype to float32 for consistency with non-cached images
            # Without this, cached images would be bfloat16 but non-cached would be float32,
            # causing batch collation to crash with "expected all tensors to be same dtype"
            if self._sidecar_dtype is torch.bfloat16:
                self._sidecar_dtype = torch.float32
                self._sidecar_dtype_str = "float32"
                self.logger.warning(
                    "cpu_bf16_cache_pipeline disabled but sidecar_storage_dtype was bfloat16. "
                    "Forcing sidecar_storage_dtype to float32 for batch consistency. "
                    "Existing bfloat16 caches will be re-processed (config hash changed)."
                )

        # Defensive validation: ensure sidecar dtype matches what non-cached processing produces
        # This prevents batch collation crashes from dtype mismatches between cached and non-cached images
        if self._sidecar_enabled:
            expected_dtype = torch.bfloat16 if self._cpu_bf16_cache else torch.float32
            if self._sidecar_dtype != expected_dtype:
                raise ValueError(
                    f"Dtype inconsistency detected: sidecar_storage_dtype={self._sidecar_dtype}, "
                    f"but cpu_bf16_cache_pipeline would produce {expected_dtype}. "
                    "This would cause batch collation failures with mixed cached/uncached images. "
                    "Either set cpu_bf16_cache_pipeline=true with sidecar_storage_dtype=bfloat16, "
                    "or set both to use float32."
                )

        # Compute config hash for sidecar cache invalidation
        # Include vocab_size to invalidate cache if vocabulary changes
        # Include has_joint_transforms to invalidate cache if transforms change
        # Include flip params to invalidate cache if flip augmentation settings change
        self._config_hash = compute_cache_config_hash(
            image_size=self.image_size,
            pad_color=self.pad_color,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            storage_dtype=self._sidecar_dtype_str,
            vocab_size=len(self.vocab.tag_to_index),
            has_joint_transforms=(self.joint_transforms is not None),
            random_flip_prob=float(random_flip_prob or 0.0),
            has_orientation_handler=(orientation_handler is not None),
        )

        # --- Orientation / flipping state ---
        self.random_flip_prob = float(random_flip_prob or 0.0)
        self.orientation_handler = orientation_handler
        self.respect_flip_list = bool(respect_flip_list)
        self._force_flip_ids: Set[str] = set()
        self._never_flip_ids: Set[str] = set()
        if flip_overrides_path:
            try:
                path = Path(flip_overrides_path)
                if path.exists():
                    data = orjson.loads(path.read_bytes()) if HAS_ORJSON else json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        force = data.get("force_flip") or data.get("flip") or []
                        never = data.get("never_flip") or data.get("no_flip") or []
                        self._force_flip_ids = {sanitize_identifier(str(x)) for x in force}
                        self._never_flip_ids = {sanitize_identifier(str(x)) for x in never}
                    elif isinstance(data, list):
                        self._force_flip_ids = {sanitize_identifier(str(x)) for x in data}
            except Exception as e:
                self.logger.warning(f"Failed to load flip_overrides from {flip_overrides_path}: {e}")

        # Telemetry (optional): push orientation stats periodically
        self._stats_queue = stats_queue
        self._samples_seen = 0

        # --- Shared vocabulary flag (for worker_init_fn) ---
        self._shared_vocab_loaded = False

        # Epoch tracking for flip variation across epochs
        # This ensures that the same image can flip/not-flip differently in different epochs
        # Default to 0 for first epoch; updated by set_epoch() in training loop
        self._current_epoch = 0
        self._epoch_was_set = False  # Track if set_epoch() was ever called
        self._epoch_warning_issued = False  # Avoid spamming warnings

        # --- Mismatch tracking for orientation tag-image consistency ---
        # Tracks when flip is applied but some orientation tags couldn't be swapped
        self._mismatch_stats: dict = {
            'force_flip_mismatches': 0,
            'random_flip_mismatches': 0,
            'mismatch_samples': [],  # Store (image_id, unmapped_tags) for debugging
        }
        self._mismatch_warning_threshold = 100  # Warn after this many mismatches
        self._mismatch_warned = False

        # --- Pre-created transforms for performance (avoid recreating per sample) ---
        # Use v2 API to avoid deprecation warning (ToTensor is deprecated)
        if T is not None:
            self._to_tensor_v2 = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
            self._to_tensor = self._to_tensor_v2  # Use v2 for v1 fallback too
        elif transforms is not None:
            self._to_tensor_v2 = None
            self._to_tensor = transforms.ToTensor()  # Legacy fallback
        else:
            self._to_tensor_v2 = None
            self._to_tensor = None

        # --- Image pre-fetching (per-worker; created lazily) ---
        # Provides 20-40% speedup by loading next images in background threads
        self._prefetcher: Optional[ImagePreFetcher] = None
        self._enable_prefetch = os.getenv("DATASET_PREFETCH", "1") != "0"

        # Pre-parse minimal fields for speed
        # items can be List[Dict] (legacy) or ArrowMetadataAccessor (zero-copy)
        self.items: Any = []
        self._using_arrow = False
        self._arrow_cache_path: Optional[Path] = None

        # Try loading from metadata cache if enabled
        if metadata_cache_enabled:
            # Try Arrow cache first (zero-copy, memory-mapped)
            from utils.metadata_cache import try_load_arrow_cache, _arrow_cache_path

            # Use prebuilt table if provided (avoids rebuilding for train/val splits)
            if prebuilt_arrow_table is not None:
                arrow_table = prebuilt_arrow_table
                self.logger.info("Using prebuilt Arrow table from parent context")
            else:
                arrow_table = try_load_arrow_cache(
                    root_dir=self.root,
                    json_files=self.json_files,
                    force_rebuild=force_rebuild_metadata_cache,
                    num_workers=metadata_cache_workers,
                    staleness_check_samples=metadata_cache_staleness_check_samples,
                    logger=self.logger
                )

            if arrow_table is not None:
                # Use ArrowMetadataAccessor for zero-copy access
                self._arrow_cache_path = _arrow_cache_path(self.root)

                # If using prebuilt table (contains ALL files), filter to this dataset's files
                if prebuilt_arrow_table is not None and "json_stem" in arrow_table.column_names:
                    # Build lookup set of (dir, stem) from our json_files
                    our_files = {(str(jp.parent), jp.stem) for jp in self.json_files}
                    dirs = arrow_table.column("dir").to_pylist()
                    stems = arrow_table.column("json_stem").to_pylist()
                    # Create boolean mask for filtering
                    mask = [(d, s) in our_files for d, s in zip(dirs, stems)]
                    arrow_table = arrow_table.filter(mask)
                    self.logger.info(f"Filtered Arrow table to {len(arrow_table):,} rows for this split")

                if self.excluded_image_ids:
                    # If there are exclusions, we need to filter them out
                    # Build a filtered list (loses zero-copy but handles exclusions correctly)
                    # This is rare - exclusions are only for bad/corrupted images
                    self.logger.info(
                        f"Filtering {len(self.excluded_image_ids)} exclusions from Arrow cache..."
                    )
                    accessor = ArrowMetadataAccessor(arrow_table, self._arrow_cache_path)
                    original_count = len(accessor)
                    filtered_items = []
                    for i in range(original_count):
                        item = accessor[i]
                        if item['image_id'] not in self.excluded_image_ids:
                            filtered_items.append(item)
                    self.items = filtered_items
                    excluded_count = original_count - len(self.items)
                    self._using_arrow = False  # Using list, not Arrow accessor
                    self.logger.info(
                        f"Filtered {excluded_count} excluded images, {len(self.items):,} items remaining "
                        "(note: zero-copy disabled due to exclusions)"
                    )
                else:
                    # No exclusions - use zero-copy Arrow accessor
                    self.items = ArrowMetadataAccessor(arrow_table, self._arrow_cache_path)
                    self._using_arrow = True
                    self.logger.info(f"Loaded {len(self.items):,} items from Arrow cache (zero-copy)")
            else:
                # Arrow cache unavailable (PyArrow not installed or build failed)
                # Fall back to sequential parsing
                self.logger.warning(
                    "Arrow metadata cache unavailable. Falling back to sequential JSON parsing. "
                    "Install PyArrow for faster loading: pip install pyarrow>=14.0.0"
                )
                metadata_cache_enabled = False  # Trigger fallback path

        # Fallback: sequential parsing (if cache disabled or failed)
        if not metadata_cache_enabled or len(self.items) == 0:
            excluded_count = 0
            for jp in self.json_files:
                try:
                    data = orjson.loads(Path(jp).read_bytes()) if HAS_ORJSON else json.loads(Path(jp).read_text(encoding="utf-8"))
                    # Skip if data is not a dict (e.g., manifest files are lists)
                    if not isinstance(data, dict):
                        self.logger.warning(f"Skipping {jp}: expected dict, got {type(data).__name__}")
                        continue
                    fname = str(data.get("filename") or jp.with_suffix(".png").name)
                    image_id = sanitize_identifier(Path(fname).stem)

                    # Skip excluded images by image_id (format-agnostic)
                    if self.excluded_image_ids and image_id in self.excluded_image_ids:
                        excluded_count += 1
                        continue

                    tags_raw = data.get("tags")
                    tags_list = parse_tags_field(tags_raw)
                    rating = data.get("rating", "unknown")
                    # Remember the shard folder this pair lives in for image resolution
                    self.items.append({
                        "image_id": image_id,
                        "tags": tags_list,
                        "rating": rating,
                        "dir": Path(jp).parent,
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to parse {jp}: {e}")
            if excluded_count > 0:
                self.logger.info(f"Filtered out {excluded_count} excluded images during parsing")

    # ---------- Pickling support for multiprocessing ----------
    def __getstate__(self):
        """Prepare for pickling - exclude unpicklable objects.

        Note: When using Arrow cache, self.items is an ArrowMetadataAccessor
        which handles its own serialization. It only pickles the cache path,
        then re-opens the memory-mapped file in each worker. This allows all
        workers to share the same physical memory pages via OS virtual memory.
        """
        state = self.__dict__.copy()
        # Remove unpicklable objects before sending to worker
        state['_prefetcher'] = None          # ImagePreFetcher thread pool (if any)
        state['orientation_handler'] = None  # May contain unpicklable state
        state['_stats_queue'] = None         # multiprocessing.Queue (cannot be pickled on Windows spawn)
        state['_exclusion_manager'] = None   # Contains threading lock (will be recreated)
        # ArrowMetadataAccessor handles its own __getstate__/__setstate__
        # It only pickles the path, then re-opens the mmap in worker
        return state

    def __setstate__(self, state):
        """Restore from pickle in worker process."""
        self.__dict__.update(state)
        # These will be lazily recreated when needed:
        # - _prefetcher and _orientation_handler if needed
        # - _stats_queue stays None in workers (telemetry only from main process)
        # - ArrowMetadataAccessor re-opens the memory-mapped file automatically

        # Recreate exclusion manager in worker process
        # This allows each worker to persist failed samples independently
        exclusion_path = self.root / 'cache_exclusions.txt'
        self._exclusion_manager = ExclusionManager(
            exclusion_path,
            reload_interval_seconds=_EXCLUSION_RELOAD_INTERVAL
        )
        # Load current exclusions (may have been updated by other workers)
        self.excluded_image_ids = self._exclusion_manager.load()

    def __len__(self) -> int:
        return len(self.items)

    # ---------- Pre-fetcher ----------
    def _ensure_prefetcher(self):
        """Create image pre-fetcher lazily (per-worker)."""
        if not self._enable_prefetch or self._prefetcher is not None:
            return
        # Configurable prefetch settings (increased defaults to hide I/O latency)
        # DATASET_PREFETCH_SIZE: Number of images to prefetch (default 32)
        # DATASET_PREFETCH_WORKERS: Number of background threads (default 4)
        # Memory overhead: ~3MB per cached image, so 32 images = ~96MB per worker
        cache_size = int(os.getenv("DATASET_PREFETCH_SIZE", "32"))
        max_workers = int(os.getenv("DATASET_PREFETCH_WORKERS", "4"))
        self._prefetcher = ImagePreFetcher(
            max_workers=max_workers, 
            cache_size=cache_size,
            target_size=self.image_size,
            pad_color=self.pad_color
        )

    def _get_image_path_for_idx(self, idx: int) -> Path:
        """Get image path for a given index (for prefetching).

        Returns:
            Path to image file

        Raises:
            Exception on any error (caught by prefetcher)
        """
        ann = self.items[idx]
        image_id = ann["image_id"]
        img_root = ann.get("dir", self.root)
        return validate_image_path(Path(img_root), image_id)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic-yet-varying flip decisions.

        This method should be called at the start of each training epoch to ensure
        that flip decisions vary across epochs while remaining deterministic for
        reproducibility.

        Args:
            epoch: Current training epoch (0-indexed)

        Note:
            - Called automatically by the training loop via DataLoader
            - Affects both training and validation datasets
            - Essential for proper cache invalidation and augmentation diversity
        """
        self._current_epoch = int(epoch)
        self._epoch_was_set = True
        self.logger.debug(f"Dataset epoch set to {self._current_epoch}")

    def _deterministic_coin(self, image_id: str) -> bool:
        """Stable per-image, per-epoch coin flip using fast CRC32 hash.

        This ensures deterministic yet epoch-varying flip decisions:
        - Same (image_id, epoch) always produces the same flip decision (reproducible)
        - Different epochs produce different flip decisions (augmentation diversity)
        - Cache-friendly: unflipped versions cached, flipped computed on-demand

        Performance: CRC32 is ~20x faster than SHA256 (~0.1μs vs ~2-5μs per call).
        At 5.6M samples/epoch, this saves ~11-28 seconds per epoch.

        Args:
            image_id: Unique image identifier

        Returns:
            True if image should be flipped in current epoch, False otherwise
        """
        if self.random_flip_prob <= 0:
            return False
        # Warn once if flip is enabled but set_epoch() was never called
        # This helps catch training loops that forget to set the epoch
        if not self._epoch_was_set and not self._epoch_warning_issued:
            self._epoch_warning_issued = True
            self.logger.warning(
                "Random flip is enabled (prob=%.2f) but set_epoch() was never called. "
                "All images will use epoch=0 for flip decisions, meaning the same images "
                "will flip the same way every epoch. Call dataset.set_epoch(epoch) at the "
                "start of each epoch for proper augmentation diversity.",
                self.random_flip_prob
            )
        # Include epoch in hash to get different flips across epochs
        # Use zlib.crc32 for speed - deterministic and fast (~20x faster than SHA256)
        seed_bytes = f"{image_id}|epoch{self._current_epoch}".encode("utf-8")
        h = zlib.crc32(seed_bytes) & 0xFFFFFFFF  # Ensure unsigned 32-bit
        v = h / 0xFFFFFFFF  # [0,1]
        return v < float(self.random_flip_prob)

    def _decide_flip_mode(self, image_id: str, tags: List[str]) -> str:
        """
        Decide flipping policy: 'none' | 'random' | 'force'
        Respects flip list first; then applies safety veto; then p-coin.
        """
        if self.respect_flip_list:
            if image_id in self._never_flip_ids:
                return "none"
            if image_id in self._force_flip_ids:
                return "force"
        if self.random_flip_prob <= 0:
            return "none"
        if self.orientation_handler is not None:
            try:
                if self.orientation_handler.should_skip_flip(tags):
                    return "none"
            except Exception:
                pass
        return "random" if self._deterministic_coin(image_id) else "none"

    def _track_flip_mismatch(
        self,
        image_id: str,
        mode: str,
        unmapped_tags: List[str]
    ) -> None:
        """Track and optionally warn about tag-image flip mismatches.

        Called when a flip is applied but some orientation-sensitive tags
        couldn't be swapped, creating a potential tag-image mismatch.

        Args:
            image_id: Image identifier for debugging
            mode: Flip mode ("force" or "random")
            unmapped_tags: Tags that needed mapping but had none
        """
        if mode == "force":
            self._mismatch_stats['force_flip_mismatches'] += 1
        else:
            self._mismatch_stats['random_flip_mismatches'] += 1

        # Store sample info for debugging (bounded list)
        if len(self._mismatch_stats['mismatch_samples']) < 50:
            self._mismatch_stats['mismatch_samples'].append({
                'image_id': image_id,
                'mode': mode,
                'unmapped_tags': unmapped_tags[:5],  # Limit stored tags
            })

        # Warn once when threshold exceeded
        total_mismatches = (
            self._mismatch_stats['force_flip_mismatches'] +
            self._mismatch_stats['random_flip_mismatches']
        )
        if not self._mismatch_warned and total_mismatches >= self._mismatch_warning_threshold:
            self._mismatch_warned = True
            sample_tags = [s['unmapped_tags'] for s in self._mismatch_stats['mismatch_samples'][:3]]
            self.logger.warning(
                f"Detected {total_mismatches} tag-image flip mismatches. "
                f"Force mode: {self._mismatch_stats['force_flip_mismatches']}, "
                f"Random mode: {self._mismatch_stats['random_flip_mismatches']}. "
                f"Sample unmapped tags: {sample_tags}. "
                f"Consider adding mappings to orientation_map.json"
            )

    def get_flip_mismatch_statistics(self) -> Dict[str, Any]:
        """Return statistics about flip-related tag mismatches.

        Useful for monitoring training quality and identifying tags
        that need orientation mappings.

        Returns:
            Dictionary with mismatch statistics including:
                - force_flip_mismatches: Count of force mode mismatches
                - random_flip_mismatches: Count of random mode mismatches
                - mismatch_samples: Sample of affected images/tags
                - total_mismatches: Total mismatch count
                - mismatch_rate: Ratio of mismatches to total flips (if available)
        """
        total_mismatches = (
            self._mismatch_stats['force_flip_mismatches'] +
            self._mismatch_stats['random_flip_mismatches']
        )

        # Calculate mismatch rate if orientation handler has flip stats
        total_flips = 0
        if self.orientation_handler is not None:
            stats = self.orientation_handler.stats
            total_flips = stats.get('total_flips', 0)

        mismatch_rate = total_mismatches / max(1, total_flips) if total_flips > 0 else 0.0

        return {
            **self._mismatch_stats,
            'total_mismatches': total_mismatches,
            'total_flips': total_flips,
            'mismatch_rate': mismatch_rate,
        }

    def _build_sample_dict(
        self,
        image: torch.Tensor,
        padding_mask: torch.Tensor,
        tag_vec: torch.Tensor,
        rating: Any,
        image_id: str,
        cached: bool = False,
        flip_applied: bool = False,
        flip_mode: str = "none",
        has_tag_mismatch: bool = False,
    ) -> Dict[str, Any]:
        """Build the sample dictionary returned by __getitem__.

        Args:
            image: Preprocessed image tensor (C, H, W)
            padding_mask: Boolean padding mask (H, W)
            tag_vec: Encoded tag vector
            rating: Rating value (to be mapped)
            image_id: Image identifier
            cached: Whether data came from cache
            flip_applied: Whether horizontal flip was applied
            flip_mode: Flip mode used ("none", "force", "random")
            has_tag_mismatch: Whether unmapped orientation tags exist (tag-image mismatch)

        Returns:
            Sample dict for training
        """
        rating_idx = _map_rating(rating)

        # Ensure tensors are contiguous before returning for efficient pin_memory
        # torch.flip() returns a view (non-contiguous), which forces implicit copies during
        # DataLoader collation/pinning. Making them contiguous here (in workers) is cheaper.
        if not image.is_contiguous():
            image = image.contiguous()
        if not padding_mask.is_contiguous():
            padding_mask = padding_mask.contiguous()

        return {
            "images": image,
            "padding_mask": padding_mask.to(torch.bool),
            "tag_labels": tag_vec,
            "rating_labels": torch.tensor(rating_idx, dtype=torch.long),
            "image_id": image_id,
            "cached": cached,
            "flip_applied": flip_applied,
            "flip_mode": flip_mode,
            "has_tag_mismatch": has_tag_mismatch,
            "error": False,
            "error_reason": "",
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Periodically reload exclusions to pick up failures from other workers
        # This runs every ~120 seconds (configurable via _EXCLUSION_RELOAD_INTERVAL)
        if self._exclusion_manager:
            if self._exclusion_manager.reload_if_stale():
                # Update local reference if new exclusions were found
                new_exclusions = self._exclusion_manager.get_excluded_ids()
                if len(new_exclusions) > len(self.excluded_image_ids):
                    self.excluded_image_ids = new_exclusions

        if idx in self.failed_samples:
            return self._error_sample(idx, "Previously failed sample")

        # Check if this sample was excluded by another worker (cross-worker sync)
        if idx < len(self.items):
            item_image_id = self.items[idx].get("image_id")
            if item_image_id and item_image_id in self.excluded_image_ids:
                # Mark as failed in memory too to speed up subsequent checks
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)
                return self._error_sample(idx, f"Excluded by other worker: {item_image_id}")

        # Track retries with memory bounds to prevent unbounded growth
        if idx not in self.retry_counts:
            # Evict oldest entries if at capacity (simple FIFO-like eviction)
            if len(self.retry_counts) >= _MAX_RETRY_COUNTS:
                # Remove ~10% of entries to amortize eviction cost
                keys_to_remove = list(self.retry_counts.keys())[:_MAX_RETRY_COUNTS // 10]
                for k in keys_to_remove:
                    del self.retry_counts[k]
            self.retry_counts[idx] = 0

        try:
            ann = self.items[idx]
            image_id = ann["image_id"]
            # Work on a copy of the tag list so we can safely modify it
            original_tags: List[str] = list(ann["tags"])
            tags_now: List[str] = original_tags
            # Decide whether to flip and adjust tags accordingly
            mode = self._decide_flip_mode(image_id, original_tags)
            flip_bit = False
            has_tag_mismatch = False
            if mode != "none" and self.orientation_handler is not None:
                # Use swap_tags_with_info() to detect potential tag-image mismatches
                swap_result = self.orientation_handler.swap_tags_with_info(
                    original_tags,
                    skip_safety_check=True,  # Safety already checked in _decide_flip_mode
                    record_stats=True
                )
                tags_now = swap_result.swapped_tags
                flip_bit = swap_result.flip_applied if mode == "random" else True
                has_tag_mismatch = swap_result.has_mismatch

                # Track mismatches for monitoring and debugging
                if has_tag_mismatch:
                    self._track_flip_mismatch(image_id, mode, swap_result.unmapped_orientation_tags)

            # Resolve image path first (needed for both cache lookup and loading)
            img_root = ann.get("dir", self.root)
            img_path = validate_image_path(Path(img_root), image_id)

            # Try sidecar cache first
            if self._sidecar_enabled:
                # Get source file mtime for cache invalidation when source is modified
                # Only fetch when sidecar cache is enabled to avoid unnecessary syscalls
                try:
                    source_mtime = os.path.getmtime(img_path)
                except OSError:
                    source_mtime = None  # File might not exist yet, will fail later
                sidecar_path = get_sidecar_path(str(img_path), self._sidecar_extension)
                cache_result = load_sidecar(
                    sidecar_path,
                    expected_config_hash=self._config_hash,
                    expected_source_mtime=source_mtime,
                )
                if cache_result is not None:
                    img_t, pmask = cache_result
                    # Verify cached shape and dtype match expected values
                    shape_ok = (img_t.dim() == 3 and
                                img_t.shape[0] == 3 and
                                img_t.shape[1] == int(self.image_size) and
                                img_t.shape[2] == int(self.image_size))
                    dtype_ok = (img_t.dtype == self._sidecar_dtype)
                    if shape_ok and dtype_ok:
                        # Apply flip transformation to cached data if needed
                        if flip_bit:
                            img_t = torch.flip(img_t, dims=[2])  # Flip width dimension (CHW format)
                            pmask = torch.flip(pmask, dims=[1])  # Flip width dimension (HW format)
                        # Use tags that already reflect the flip decision
                        tag_vec = self.vocab.encode_tags(tags_now)
                        monitor.l2_hit()  # Sidecar cache hit
                        return self._build_sample_dict(
                            img_t, pmask, tag_vec, ann.get("rating", "unknown"), image_id,
                            cached=True,
                            flip_applied=flip_bit,
                            flip_mode=mode,
                            has_tag_mismatch=has_tag_mismatch,
                        )
                    else:
                        # Cache file loaded but validation failed - this is a stale entry, not a miss
                        if self.logger.isEnabledFor(logging.DEBUG):
                            if not shape_ok:
                                self.logger.debug(f"Sidecar shape mismatch for {image_id}, reloading")
                            elif not dtype_ok:
                                self.logger.debug(f"Sidecar dtype mismatch for {image_id}: {img_t.dtype} != {self._sidecar_dtype}")
                        monitor.l2_stale()  # Stale cache entry (loaded but invalid)
                else:
                    monitor.l2_miss()  # True cache miss (file not found or couldn't load)

            # Cache miss: load from disk
            # Try to use pre-fetched image first (reduces I/O latency by ~20-40%)
            self._ensure_prefetcher()
            prefetch_result = None
            if self._prefetcher is not None:
                prefetch_result = self._prefetcher.get(idx)
                # Trigger lookahead for next images
                self._prefetcher.trigger_lookahead(
                    idx, len(self), self._get_image_path_for_idx
                )

            # If not pre-fetched, load synchronously
            if prefetch_result is None:
                # Fully decode and correct EXIF while file is open
                with Image.open(img_path) as pil_img:
                    pil_img.load()
                    pil_img = ImageOps.exif_transpose(pil_img)
                    if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                        rgba = pil_img.convert("RGBA")
                        bg = Image.new("RGB", rgba.size, self.pad_color)
                        alpha = rgba.getchannel("A")
                        bg.paste(rgba, mask=alpha)
                        pil = bg
                    else:
                        pil = pil_img.convert("RGB")
                
                canvas, pmask = process_image_cpu(pil, self.image_size, self.pad_color)
            else:
                canvas, pmask = prefetch_result

            # NOTE: Flip is applied AFTER joint_transforms to ensure correct ordering
            # (transforms operate on canonical unflipped images, flip is applied last)
            # Joint v2 transforms keep geometry aligned with mask when used
            if self.joint_transforms is not None and T is not None and tv_tensors is not None:
                img_tv = tv_tensors.Image(canvas)
                mask_tv = tv_tensors.Mask(pmask.to(torch.uint8))
                img_tv, mask_tv = self.joint_transforms(img_tv, mask_tv)
                img = self._to_tensor_v2(img_tv)
                if self._cpu_bf16_cache:
                    img = img.to(torch.bfloat16)
                img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)
                pmask = mask_tv.to(torch.bool)
            else:
                # Fallback: color-only transforms permitted
                if self.transform:
                    try:
                        transformed = self.transform(canvas)
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img = transformed if isinstance(transformed, torch.Tensor) else self._to_tensor(transformed)
                        if self._cpu_bf16_cache:
                            img = img.to(torch.bfloat16)
                        img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)
                    except Exception as e:
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Transform failed, using fallback: {e}")
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img = self._to_tensor(canvas)
                        if self._cpu_bf16_cache:
                            img = img.to(torch.bfloat16)
                        img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)
                else:
                    if self._to_tensor is None:
                        raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                    img = self._to_tensor(canvas)
                    if self._cpu_bf16_cache:
                        img = img.to(torch.bfloat16)
                    img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)

            # Encode labels (tags already account for flipping)
            tag_vec = self.vocab.encode_tags(tags_now)  # (V,)

            # Save sidecar cache file BEFORE applying flip (stores canonical unflipped version)
            if self._sidecar_enabled:
                # Store the UNFLIPPED version directly - flip will be applied after caching
                img_to_cache = img
                mask_to_cache = pmask

                # Use file locking to reduce redundant I/O from concurrent writes
                # Skip locking for very long paths to avoid filesystem limits (Windows: 260 chars)
                # Note: save_sidecar() uses atomic writes (UUID temp + os.replace), so multiple
                # writes are safe even without locking - locking just reduces wasted work
                use_locking = HAS_FILELOCK and len(sidecar_path) <= 240  # Leave room for .lock extension
                if use_locking:
                    lock_path = sidecar_path + ".lock"
                    try:
                        with filelock.FileLock(lock_path, timeout=0.1):  # Non-blocking attempt
                            # Got lock - write the cache file
                            # Skip existence check to avoid TOCTOU race; atomic write handles conflicts
                            save_sidecar(
                                sidecar_path,
                                img_to_cache.to(self._sidecar_dtype),
                                mask_to_cache,
                                self._config_hash,
                                image_size=self.image_size,
                                source_mtime=source_mtime,
                            )
                    except filelock.Timeout:
                        pass  # Another worker is writing, skip to avoid redundant work
                else:
                    # Without locking, still safe due to UUID-based temp files in save_sidecar
                    save_sidecar(
                        sidecar_path,
                        img_to_cache.to(self._sidecar_dtype),
                        mask_to_cache,
                        self._config_hash,
                        image_size=self.image_size,
                        source_mtime=source_mtime,
                    )

            # Apply horizontal flip AFTER caching (ensures correct transform ordering)
            # Flip is applied at tensor level after all other transforms complete
            if flip_bit:
                img = torch.flip(img, dims=[2])  # Flip width dimension (CHW format)
                pmask = torch.flip(pmask, dims=[1])  # Flip width dimension (HW format)

            self.retry_counts[idx] = 0
            return self._build_sample_dict(
                img, pmask, tag_vec, ann.get("rating", "unknown"), image_id,
                cached=False,
                flip_applied=flip_bit,
                flip_mode=mode,
                has_tag_mismatch=has_tag_mismatch,
            )

        except Exception as e:
            self.retry_counts[idx] += 1
            self._sample_error_log_count += 1
            # Rate-limit warning logs: log first, then every 100th
            if self._sample_error_log_count == 1 or self._sample_error_log_count % 100 == 0:
                self.logger.warning(f"Failed to load sample {idx}: {e} (total errors: {self._sample_error_log_count})")
            if self.retry_counts[idx] >= self.max_retries:
                # Add to failed set with memory bounds
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)

                # Persist failed sample to exclusion file immediately
                # This ensures the sample is skipped in future epochs/runs
                try:
                    failed_image_id = self.items[idx].get("image_id") if idx < len(self.items) else None
                    if failed_image_id and self._exclusion_manager:
                        was_new = self._exclusion_manager.add_exclusion(failed_image_id, immediate=True)
                        if was_new:
                            self.excluded_image_ids.add(failed_image_id)
                            self.logger.info(
                                f"Persisted exclusion for {failed_image_id} (sample {idx}) - "
                                f"will be skipped in future runs"
                            )
                except Exception as persist_err:
                    self.logger.warning(f"Could not persist exclusion for sample {idx}: {persist_err}")

                # Always log when sample permanently fails (rate-limited by max_retries)
                self.logger.error(f"Sample {idx} exceeded max retries, marking as failed")
                return self._error_sample(idx, str(e))
            return self._error_sample(idx, f"Temporary failure: {e}")
        finally:
            # Opportunistic telemetry: push orientation stats every 128 samples
            try:
                if self._stats_queue is not None and self.orientation_handler is not None:
                    self._samples_seen += 1
                    if (self._samples_seen & 127) == 0:
                        stats = self.orientation_handler.get_statistics()
                        payload = {
                            "flip_total": int(stats.get("total_flips", 0)),
                            "flip_safe": int(stats.get("safe_flips", 0)),
                            "flip_skipped_text": int(stats.get("blocked_by_text", 0)),
                            "flip_skipped_unmapped": int(stats.get("skipped_flips", 0)),
                            "flip_blocked_safety": int(stats.get("blocked_by_safety", 0)),
                        }
                        self._stats_queue.put_nowait(payload)
            except Exception:
                pass

    def _error_sample(self, idx: int, reason: str) -> Dict[str, Any]:
        # Always use self.image_size for consistency with actual samples
        # This ensures error samples have the same shape as valid samples for batching
        sz = self.image_size  # Already int from __init__
        # Match image dtype to what cached/non-cached samples use to avoid batch collation issues
        # Use _sidecar_dtype directly since that's what both paths now produce
        img_dtype = self._sidecar_dtype
        return {
            "images": torch.zeros((3, sz, sz), dtype=img_dtype),
            "padding_mask": torch.ones((sz, sz), dtype=torch.bool),
            "tag_labels": torch.zeros(len(self.vocab.tag_to_index), dtype=self._tag_vector_dtype),
            "rating_labels": torch.tensor(4, dtype=torch.long),
            "image_id": f"error_{idx}",
            "cached": False,
            "flip_applied": False,
            "flip_mode": "none",
            "has_tag_mismatch": False,
            "error": True,
            "error_reason": reason,
        }


def _map_rating(rating: Any) -> int:
    """Map dataset rating field to fixed indices used by the model.

    Mapping:
        - general/safe/g -> 0
        - sensitive -> 1
        - questionable/q -> 2
        - explicit/e -> 3
        - unknown/u -> 4 (default)

    Args:
        rating: Rating value from dataset (int or str)

    Returns:
        Integer rating index (0-4)
    """
    if isinstance(rating, int):
        # Validate range
        idx = int(rating)
        if 0 <= idx <= 4:
            return idx
        else:
            logging.getLogger(__name__).warning(
                f"Invalid rating index {idx}, defaulting to 4 (unknown)"
            )
            return 4

    r = str(rating).strip().lower()
    mapping = {
        "g": 0, "general": 0, "safe": 0,
        "sensitive": 1,
        "q": 2, "questionable": 2,
        "e": 3, "explicit": 3,
        "u": 4, "unknown": 4,
    }
    return mapping.get(r, 4)


def create_dataloaders(
    data_config,
    validation_config,
    vocab_path,
    active_data_path,
    distributed=False,
    rank=-1,
    world_size=1,
    seed=42,
    debug_config=None,
    **kwargs,
):
    logger = logging.getLogger(__name__)

    # Extract config once to avoid redundant processing
    config_cache = {
        # Sidecar cache configuration (replaces L2 LMDB cache)
        'sidecar_cache_enabled': bool(getattr(data_config, "sidecar_cache_enabled", True)),
        'sidecar_extension': str(getattr(data_config, "sidecar_extension", ".safetensor")),
        'sidecar_storage_dtype': str(getattr(data_config, "sidecar_storage_dtype", "bfloat16")),
        'preload_files': int(getattr(data_config, "preload_files", 0)),
        'cpu_bf16_cache_pipeline': getattr(data_config, "cpu_bf16_cache_pipeline", None),
        # Image processing configuration
        'image_size': int(getattr(data_config, "image_size", 512)),
        'normalize_mean': tuple(getattr(data_config, "normalize_mean", [0.5, 0.5, 0.5])),
        'normalize_std': tuple(getattr(data_config, "normalize_std", [0.5, 0.5, 0.5])),
        'pad_color': tuple(getattr(data_config, "pad_color", [114, 114, 114])),
        # Orientation/flip configuration
        'random_flip_prob': float(getattr(data_config, "random_flip_prob", 0.0)),
        'orientation_map_path': getattr(data_config, "orientation_map_path", None),
        'flip_overrides_path': getattr(data_config, "flip_overrides_path", None),
        'strict_orientation_validation': bool(getattr(data_config, "strict_orientation_validation", False)),
        'orientation_safety_mode': str(getattr(data_config, "orientation_safety_mode", "conservative")),
        'skip_unmapped': bool(getattr(data_config, "skip_unmapped", False)),
        'stats_queue': getattr(data_config, "stats_queue", None),
        # DataLoader configuration
        'drop_last': bool(getattr(data_config, "drop_last", False)),
        # Metadata cache configuration
        'metadata_cache_enabled': bool(getattr(data_config, "metadata_cache_enabled", True)),
        'metadata_cache_workers': int(getattr(data_config, "metadata_cache_workers", 16)),
        'force_rebuild_metadata_cache': bool(getattr(data_config, "force_rebuild_metadata_cache", False)),
        'metadata_cache_staleness_check_samples': int(getattr(data_config, "metadata_cache_staleness_check_samples", 100)),
        # Validation split limiting
        'max_val_samples': getattr(data_config, "max_val_samples", None),
    }

    if config_cache['sidecar_cache_enabled']:
        logger.info(
            "Sidecar cache enabled (extension=%s, dtype=%s)",
            config_cache['sidecar_extension'],
            config_cache['sidecar_storage_dtype'],
        )
    else:
        logger.info("Sidecar cache disabled")

    # Load vocabulary once (needed for sidecar mode and to determine num classes)
    vocab = load_vocabulary_for_training(Path(vocab_path))
    num_tags = len(vocab.tag_to_index)

    # Create shared vocabulary to avoid pickling overhead with spawn multiprocessing
    # This reduces worker startup time from ~400ms (8 workers × 50ms) to near-zero
    shared_vocab_manager = None
    shared_vocab_info = None
    if is_shared_memory_available():
        try:
            shared_vocab_manager = SharedVocabularyManager()
            shm_name = shared_vocab_manager.create_from_vocab(vocab)
            shared_vocab_info = (shm_name, shared_vocab_manager.vocab_size)
            logger.info(f"Created shared vocabulary ({shared_vocab_manager.vocab_size / 1024:.1f} KB)")
            # Register cleanup on program exit
            atexit.register(shared_vocab_manager.cleanup)
        except Exception as e:
            logger.warning(f"Failed to create shared vocabulary, falling back to pickling: {e}")
            shared_vocab_manager = None
            shared_vocab_info = None
    else:
        logger.debug("Shared memory not available (requires Python 3.8+), using vocabulary pickling")

    image_size = config_cache['image_size']
    mean = config_cache['normalize_mean']
    std = config_cache['normalize_std']
    pad_color = config_cache['pad_color']
    transform = None

    # Determine dataset mode
    root = Path(active_data_path)
    manifest_train = root / "train.json"
    manifest_val = root / "val.json"
    images_dir = root / "images"

    # Orientation handler / flip list wiring
    random_flip_prob = config_cache['random_flip_prob']
    orientation_map_path = config_cache['orientation_map_path']
    if isinstance(orientation_map_path, str) and orientation_map_path:
        orientation_map_path = Path(orientation_map_path)
    flip_overrides_path = config_cache['flip_overrides_path']
    _handler = None
    try:
        if random_flip_prob > 0 and OrientationHandler is not None:
            _handler = OrientationHandler(
                mapping_file=orientation_map_path if orientation_map_path else None,
                random_flip_prob=random_flip_prob,
                strict_mode=config_cache['strict_orientation_validation'],
                safety_mode=config_cache['orientation_safety_mode'],
                skip_unmapped=config_cache['skip_unmapped'],
            )
    except Exception as e:
        logger.warning(f"OrientationHandler init failed; flips disabled: {e}")

    if manifest_train.exists() and manifest_val.exists() and images_dir.exists():
        # Manifest mode (back-compat)
        # Manifest datasets are not orientation-aware. If flips are enabled, warn and disable them.
        if float(getattr(data_config, "random_flip_prob", 0.0) or 0.0) > 0.0:
            logger.warning(
                "random_flip_prob > 0 with manifest dataset; disabling orientation-aware flips (manifest is non-orientation-aware)."
            )
            try:
                setattr(data_config, "random_flip_prob", 0.0)
            except Exception:
                pass
        # Note: Manifest mode uses legacy DatasetLoader which doesn't support sidecar cache.
        # Caching is disabled for manifest mode. Migrate to sidecar JSON mode for caching support.
        logger.warning(
            "Manifest mode detected. Sidecar caching not supported for manifest datasets. "
            "Consider migrating to per-image JSON sidecar files for caching support."
        )
        train_ds = DatasetLoader(
            annotations_path=str(manifest_train),
            image_dir=str(images_dir),
            dataset_root=str(root),
            transform=transform,
            num_classes=num_tags,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            preload_files=config_cache['preload_files'],
        )

        val_ds = DatasetLoader(
            annotations_path=str(manifest_val),
            image_dir=str(images_dir),
            dataset_root=str(root),
            transform=transform,
            num_classes=num_tags,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            preload_files=config_cache['preload_files'],
        )
    else:
        # Sidecar JSON mode: scan per-image *.json recursively (shard-aware)
        logger.info("Manifest not found; entering sidecar JSON mode (scanning .json next to images)")

        cached = _try_load_cached_split(root, seed=int(seed))
        if cached is not None:
            train_list, val_list = cached
        else:
            all_jsons = sorted(root.rglob("*.json")) if root.exists() else []
            all_jsons_before_filter = len(all_jsons)

            # Exclude manifest files from sidecar parsing (uses _EXCLUSION_PATTERNS constant)
            all_jsons = [jp for jp in all_jsons if jp.name not in _EXCLUSION_PATTERNS]

            if not all_jsons:
                if all_jsons_before_filter > 0:
                    # Found JSONs but they were all manifests
                    raise FileNotFoundError(
                        f"Found {all_jsons_before_filter} JSON file(s) under {root}, but they were all "
                        f"manifest files ({', '.join(_EXCLUSION_PATTERNS)}). Expected per-image JSON sidecars. "
                        f"If you have a manifest-based dataset, place {_EXCLUSION_PATTERNS[0]} and {_EXCLUSION_PATTERNS[1]} "
                        f"directly in {root} (not subdirectories) along with an images/ directory to use manifest mode."
                    )
                else:
                    raise FileNotFoundError(
                        f"No annotation JSON files found under {root}. Expected per-image JSON sidecars."
                    )

            # Deterministic split
            import random as _random
            rng = _random.Random(int(seed))
            rng.shuffle(all_jsons)
            split_ratio = 0.95
            n_train = max(1, int(len(all_jsons) * split_ratio))
            # Ensure validation set doesn't overlap with training set
            # If we only have 1 sample, train on it and validation will be empty
            # (downstream code should handle empty validation gracefully)
            if n_train >= len(all_jsons) and len(all_jsons) > 1:
                # Keep at least 1 sample for validation when possible
                n_train = len(all_jsons) - 1
            train_list = all_jsons[:n_train]
            val_list = all_jsons[n_train:]
            _write_cached_split(root, train_list, val_list, seed=int(seed))

        # Limit validation samples at split time if configured
        # Excess validation samples are moved to training (not discarded)
        max_val_samples = config_cache['max_val_samples']
        if max_val_samples and len(val_list) > max_val_samples:
            original_val_size = len(val_list)
            excess_val = val_list[max_val_samples:]
            val_list = val_list[:max_val_samples]
            train_list = train_list + excess_val  # Move excess to training
            logger.info(
                f"Validation limited to {max_val_samples:,} samples at split time "
                f"(was {original_val_size:,}, moved {len(excess_val):,} to training)"
            )

        # Build Arrow metadata cache ONCE from ALL files (train + val combined)
        # This ensures warmup and training share the same complete cache.
        # Individual datasets will filter to their subset.
        prebuilt_arrow_table = None
        if config_cache['metadata_cache_enabled']:
            all_jsons_combined = train_list + val_list  # Full dataset
            from utils.metadata_cache import try_load_arrow_cache
            logger.info(f"Building/loading Arrow cache from {len(all_jsons_combined):,} total files...")
            prebuilt_arrow_table = try_load_arrow_cache(
                root_dir=root,
                json_files=all_jsons_combined,
                force_rebuild=config_cache['force_rebuild_metadata_cache'],
                num_workers=config_cache['metadata_cache_workers'],
                staleness_check_samples=config_cache['metadata_cache_staleness_check_samples'],
                logger=logger,
            )
            if prebuilt_arrow_table is not None:
                logger.info(f"Arrow cache ready: {len(prebuilt_arrow_table):,} rows")

        train_ds = SidecarJsonDataset(
            root_dir=root,
            json_files=train_list,
            vocab=vocab,
            transform=transform,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            sidecar_cache_enabled=config_cache['sidecar_cache_enabled'],
            sidecar_extension=config_cache['sidecar_extension'],
            sidecar_storage_dtype=config_cache['sidecar_storage_dtype'],
            cpu_bf16_cache_pipeline=config_cache['cpu_bf16_cache_pipeline'],
            random_flip_prob=random_flip_prob,
            orientation_handler=_handler,
            flip_overrides_path=flip_overrides_path,
            stats_queue=config_cache['stats_queue'],
            metadata_cache_enabled=config_cache['metadata_cache_enabled'],
            metadata_cache_workers=config_cache['metadata_cache_workers'],
            force_rebuild_metadata_cache=False,  # Already built above
            metadata_cache_staleness_check_samples=config_cache['metadata_cache_staleness_check_samples'],
            prebuilt_arrow_table=prebuilt_arrow_table,
        )

        val_ds = SidecarJsonDataset(
            root_dir=root,
            json_files=val_list,
            vocab=vocab,
            transform=transform,
            image_size=image_size,
            pad_color=pad_color,
            normalize_mean=mean,
            normalize_std=std,
            sidecar_cache_enabled=config_cache['sidecar_cache_enabled'],
            sidecar_extension=config_cache['sidecar_extension'],
            sidecar_storage_dtype=config_cache['sidecar_storage_dtype'],
            cpu_bf16_cache_pipeline=config_cache['cpu_bf16_cache_pipeline'],
            random_flip_prob=0.0,          # keep val deterministic
            orientation_handler=_handler,  # still needed to encode swapped tags if you ever TTA
            flip_overrides_path=None,
            stats_queue=config_cache['stats_queue'],
            # Now sharing prebuilt cache properly - no file count mismatch
            metadata_cache_enabled=config_cache['metadata_cache_enabled'],
            metadata_cache_workers=config_cache['metadata_cache_workers'],
            force_rebuild_metadata_cache=False,  # Already built above
            metadata_cache_staleness_check_samples=config_cache['metadata_cache_staleness_check_samples'],
            prebuilt_arrow_table=prebuilt_arrow_table,
        )

    # Samplers for distributed training
    # Always use DistributedSampler (with world_size=1 if not distributed)
    # This ensures deterministic shuffling per epoch that is independent of global RNG state,
    # which is critical for correct mid-epoch resumption.
    #
    # ResumableSampler extends DistributedSampler with O(1) mid-epoch resume support
    # by allowing direct offset into the shuffled indices instead of iterating through.
    train_sampler = ResumableSampler(
        train_ds,
        num_replicas=int(world_size) if distributed else 1,
        rank=int(rank) if distributed else 0,
        shuffle=True,
        drop_last=config_cache['drop_last'],
        seed=int(seed) if seed is not None else 0,
    )
    
    val_sampler = None
    if distributed:
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=False,
            drop_last=False,
        )
    # --------------------------------------------------------------------

    # DataLoaders
    def _dl_kwargs(cfg, *, shuffle: bool, drop_last: bool):
        kw = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=getattr(cfg, "pin_memory", False),
            drop_last=drop_last,
            shuffle=shuffle,
        )
        # Only use multiprocessing knobs when workers > 0
        if int(getattr(cfg, "num_workers", 0) or 0) > 0:
            if getattr(cfg, "prefetch_factor", None) is not None:
                kw["prefetch_factor"] = cfg.prefetch_factor
            kw["persistent_workers"] = bool(getattr(cfg, "persistent_workers", False))
        return kw

    _train_kw = _dl_kwargs(
        data_config,
        shuffle=(train_sampler is None),
        drop_last=config_cache['drop_last'],
    )
    if train_sampler is not None:
        _train_kw["sampler"] = train_sampler
    # Attach logging QueueHandler in workers if a queue is provided
    log_queue = kwargs.get("log_queue")
    _train_kw["worker_init_fn"] = WorkerInitializer(log_queue, shared_vocab_info)
    train_loader = DataLoader(train_ds, **_train_kw)

    val_batch = (
        validation_config.dataloader.batch_size
        if hasattr(validation_config, "dataloader")
        else data_config.batch_size
    )
    # Build kwargs for val loader separately to honor val batch size
    _val_kw = _dl_kwargs(data_config, shuffle=False, drop_last=False)
    _val_kw["batch_size"] = val_batch
    if val_sampler is not None:
        _val_kw["sampler"] = val_sampler
    _val_kw["worker_init_fn"] = WorkerInitializer(log_queue, shared_vocab_info)
    val_loader = DataLoader(val_ds, **_val_kw)

    # Verify pin_memory is enabled for GPU training (Critical for non_blocking transfers)
    # Without pin_memory, non_blocking=True in .to(device) has no effect
    if train_loader.pin_memory:
        logger.info("DataLoader pin_memory enabled - async H2D transfers active")
    else:
        logger.warning(
            "DataLoader pin_memory is DISABLED. This significantly degrades GPU utilization. "
            "Set data.pin_memory=true in config for optimal performance with non_blocking transfers."
        )

    # Validate datasets have samples (first-time startup safety check)
    if len(train_ds) == 0:
        raise ValueError(
            "Training dataset has 0 samples. Check that:\n"
            "  1. Annotation files (train.json or *.json sidecars) contain entries\n"
            "  2. Image files exist in the expected locations\n"
            f"  3. Data path is correct: {root}"
        )
    if len(val_ds) == 0:
        logger.warning(
            "Validation dataset has 0 samples. Validation will be skipped. "
            "This typically happens with very small datasets (1-2 samples) where "
            "all samples are allocated to training. To enable validation, add more samples."
        )
        # Return None for val_loader to signal callers to skip validation
        # This prevents downstream errors like division by zero in metrics
        val_loader = None

    return train_loader, val_loader, vocab
