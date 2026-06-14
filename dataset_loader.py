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
- Metadata Cache: Parsed JSON metadata cached as Arrow IPC for zero-copy memory-mapped access

VOCABULARY:
- Built automatically by vocabulary.py:create_vocabulary_from_datasets()
- Scans all JSON files, counts tag frequencies, saves to vocabulary.json
- See vocabulary.py for details

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
from collections import OrderedDict, defaultdict
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

# PyArrow for zero-copy metadata cache (memory-mapped, shared across workers)
try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import pyarrow.compute as pc  # Vectorized filtering operations
    HAS_PYARROW = True
except ImportError:
    pa = None
    pa_ipc = None
    pc = None
    HAS_PYARROW = False

import logging.handlers
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

# Third-party imports
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset, DataLoader as _TorchDataLoader
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
from utils.metadata_ingestion import parse_tags_field
# Safetensors fallback removed - Arrow is now the only metadata cache format
from utils.path_utils import sanitize_identifier, validate_image_path
from utils.exclusion_manager import ExclusionManager
from vocabulary import load_vocabulary_for_training, TagVocabulary


def _nested_error_counter() -> "defaultdict[str, int]":
    """Factory for per-tag error stat counters.

    Module-level (rather than a lambda) so DatasetLoader instances stay
    picklable for DataLoader workers under Windows spawn.
    """
    return defaultdict(int)


class IndependentColorJitter:
    """
    Color jitter with independent per-parameter probability.
    Unlike torchvision.transforms.ColorJitter which applies all parameters together,
    this applies each transformation independently based on its own probability.

    Optimized to:
    - Pre-compute random decisions in batch to reduce random.random() calls
    - Short-circuit early when no transforms are active
    - Store parameters for potential combined transform creation
    """

    def __init__(
        self,
        brightness: float = 0.1,
        brightness_p: float = 0.15,
        contrast: float = 0.1,
        contrast_p: float = 0.15,
        saturation: float = 0.1,
        saturation_p: float = 0.15,
    ):
        # Store probabilities as tuple for batch random comparison
        self._probs = (brightness_p, contrast_p, saturation_p)
        # Pre-create individual transforms for efficiency
        # Store as tuple for indexed access (faster than dict)
        if T is not None:
            self._transforms = (
                T.ColorJitter(brightness=brightness) if brightness > 0 else None,
                T.ColorJitter(contrast=contrast) if contrast > 0 else None,
                T.ColorJitter(saturation=saturation) if saturation > 0 else None,
            )
            # Pre-compute which transforms are available (avoid None checks in hot path)
            self._active_indices = tuple(i for i, t in enumerate(self._transforms) if t is not None)
            # Early exit flag: if no transforms configured, __call__ can return immediately
            self._has_any_transform = len(self._active_indices) > 0
        else:
            self._transforms = (None, None, None)
            self._active_indices = ()
            self._has_any_transform = False

    def __call__(self, img):
        # Fast path: no transforms configured
        if not self._has_any_transform:
            return img

        # Batch random decision: generate all random values at once
        # This is faster than 3 separate random.random() calls due to reduced function call overhead
        rand_vals = (random.random(), random.random(), random.random())

        # Determine which transforms to apply based on pre-computed active indices
        # Only iterate over transforms that are actually configured (not None)
        apply_mask = tuple(
            rand_vals[i] < self._probs[i] for i in self._active_indices
        )

        # Fast path: no transforms selected this call
        if not any(apply_mask):
            return img

        # Apply only the selected transforms
        for idx, should_apply in zip(self._active_indices, apply_mask):
            if should_apply:
                img = self._transforms[idx](img)

        return img

# Pillow resampling compatibility and truncated image handling
try:  # Pillow ≥10
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # Pillow <10
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_LANCZOS = Image.LANCZOS

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
# Set to 300s to reduce lock contention during training (exclusions change rarely)
_EXCLUSION_RELOAD_INTERVAL = 300.0

# Sample interval for exclusion staleness checks - avoids calling reload_if_stale()
# on every single sample access. Check once per batch (64 samples) is sufficient.
_EXCLUSION_CHECK_SAMPLE_INTERVAL = 64

# Minimal dtype mapping for cache plumbing
_DTYPE_MAP = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def _canon_dtype(s: str) -> torch.dtype:
    return _DTYPE_MAP.get(str(s).lower(), torch.bfloat16)

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

    Args:
        img: Source PIL Image (RGB). Letterboxing happens in RGB-space; the
            caller is responsible for any subsequent channel reordering to
            match the configured ``color_order`` before normalization.
        target_size: Target dimension (square)
        pad_color: Symmetric (114,114,114) tuple for padding (channel-order
            agnostic).

    Returns:
        (canvas, pmask): Processed PIL Image (still RGB) and boolean padding
        mask. The output channel order downstream is determined by the
        dataset's ``color_order`` setting.
    """
    w, h = img.size
    # Downscale-only letterbox: preserve aspect, never upscale
    ratio = min(target_size / float(w), target_size / float(h)) if (w > 0 and h > 0) else 1.0
    scale = min(1.0, ratio)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    # LANCZOS (windowed-sinc) for the downscale: higher frequency response near
    # Nyquist than bilinear, which blurs+aliases below Nyquist and discards
    # fine line-art detail (eye highlights, single hair strands) before it ever
    # reaches the patch tokens. This is a downscale-only path (scale <= 1.0), so
    # there is no upscale ringing. See todos/progressive-training-plan.md v2
    # "fix the resampler" note. NOTE: this shifts the input distribution — match
    # the inference/serving preprocessor to LANCZOS for any model trained after
    # this change to avoid train/serve skew.
    resized = img.resize((max(1, nw), max(1, nh)), RESAMPLE_LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    left = (target_size - resized.size[0]) // 2
    top = (target_size - resized.size[1]) // 2
    canvas.paste(resized, (left, top))

    pmask = torch.ones(target_size, target_size, dtype=torch.bool)
    pmask[top:top + resized.size[1], left:left + resized.size[0]] = False
    
    return canvas, pmask


def apply_random_rotation(
    canvas: Image.Image,
    min_degrees: float,
    max_degrees: float,
    pad_color: Tuple[int, int, int],
) -> Image.Image:
    """Rotate canvas by a random angle from [-max,-min] ∪ [+min,+max].

    Padding mask is intentionally NOT rotated — corner pixels filled with
    pad_color follow f(θ) = tan(θ/2)·(1−tan(θ/2))/(1+tan(θ/2)), e.g. ~6.1% at
    8° and ~10.1% at 15°, well within the letterbox padding distribution the
    model already learns to ignore.

    Uses bicubic resampling to better preserve fine-grained features (eye
    highlights, hair strands, accessories). Per Thévenaz/Blu/Unser (2000,
    IEEE TIP), bicubic frequency response stays >0.85 at Nyquist for typical
    sub-pixel offsets vs. ~0.64 for bilinear.
    """
    angle = random.uniform(min_degrees, max_degrees)
    if random.random() < 0.5:
        angle = -angle
    return canvas.rotate(angle, resample=RESAMPLE_BICUBIC, expand=False, fillcolor=pad_color)


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
        saved_size = state.get('total_size', self.total_size)
        if saved_size != self.total_size:
            logging.getLogger(__name__).warning(
                "Dataset size changed from %d to %d since checkpoint was saved. "
                "Mid-epoch sample offset cannot be applied safely — resuming from epoch start.",
                saved_size, self.total_size
            )
            self._start_index = 0
            self.set_epoch(state.get('epoch', self.epoch))
            return
        self.set_epoch(state['epoch'])
        self._start_index = state.get('start_index', 0)

    def __iter__(self):
        # Generate indices using parent's logic
        indices = list(super().__iter__())

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
      - Stratified validation: sample 300 paths (150 train + 150 val) from beginning/end/middle instead of the full list
      - Parallel existence checks using ThreadPoolExecutor
      - Early return on cache hit without full validation

    Validation:
      - Version check
      - Exclusion hash check (detects changes to manifest file filtering)
      - Cache internal consistency check (header FILE_COUNT matches path count)
      - Seed check (detects when seed changes between runs)
      - Sampled existence check (verifies a stratified subset of cached paths
        still exists, modeled on the Arrow cache's sampled staleness check).
        NOTE: a full-filesystem rglob count was removed here — it walked all
        ~5.6M files on every startup, costing nearly as much as the scan the
        cache avoids. Newly added dataset files are NOT auto-detected; delete
        the split cache (logs/splits/) or change the seed to force a re-scan.
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

            # Sample 150 from train, 150 from val (300 total) — a cheap probe
            # that replaces the former full-filesystem rglob count, which
            # re-walked all ~5.6M files on every startup.
            sample_paths.extend(stratified_sample(train_list, 150))
            sample_paths.extend(stratified_sample(val_list, 150))

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
        # Atomic write pattern: write to temp then replace
        # Use os.replace() instead of Path.rename() — on Windows, rename() raises
        # WinError 183 if the destination already exists; os.replace() overwrites atomically.
        import os as _os
        train_tmp = train_file.with_suffix(".tmp")
        val_tmp = val_file.with_suffix(".tmp")

        # Remove any stale .tmp files left by a previously interrupted write
        for tmp in (train_tmp, val_tmp):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

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

        _os.replace(train_tmp, train_file)
        _os.replace(val_tmp, val_file)

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

    When pickled for multiprocessing, only the cache path and the selected
    row indices are serialized. Workers re-open the memory-mapped file
    independently (sharing physical pages via the OS) and re-apply the row
    selection so they serve the exact same rows as the main process.

    Memory savings: ~15 GB per worker for 5.6M images dataset.
    """

    def __init__(self, table: "pa.Table", cache_path: Path,
                 row_indices: Optional["np.ndarray"] = None):
        """Initialize accessor.

        Args:
            table: PyArrow Table — must already be filtered to ``row_indices``
                (i.e. ``full_table.take(row_indices)``) when indices are given
            cache_path: Path to the Arrow IPC file (for pickling)
            row_indices: Row indices into the FULL on-disk cache that this
                accessor represents (split/exclusion/bad-row filters), or None
                when the table is the unfiltered on-disk cache. CRITICAL for
                correctness: without these, workers reloading the combined
                train+val cache from disk would silently serve unfiltered rows
                (train/val leakage).
        """
        self._table = table
        self._cache_path = cache_path
        self._row_indices = row_indices
        self._len = len(table)

    def __len__(self) -> int:
        return self._len

    def _ensure_table(self) -> None:
        """Lazy-load the Arrow table on first access.

        This defers the memory-map open until the worker actually needs data,
        allowing all workers to spawn in parallel without blocking on I/O.
        The memory-map operation itself is fast (~10ms), but doing it during
        __setstate__ serializes worker startup on Windows (spawn context).
        """
        if self._table is None:
            from utils.metadata_cache import _load_arrow_cache
            import logging
            logger = logging.getLogger(__name__)
            table = _load_arrow_cache(self._cache_path, logger)
            if table is None:
                raise RuntimeError(
                    f"Failed to load Arrow metadata cache: {self._cache_path}. "
                    "The cache file may be missing, corrupted, or locked. "
                    "Try deleting the cache file and restarting training."
                )
            # Re-apply the row selection computed in the main process — the
            # on-disk cache contains ALL rows (train+val, pre-exclusion).
            # Skipping this would alias the splits and leak val into train.
            if self._row_indices is not None:
                table = table.take(self._row_indices)
            if len(table) != self._len:
                raise RuntimeError(
                    f"Arrow metadata cache row count mismatch after reload: "
                    f"expected {self._len}, got {len(table)} ({self._cache_path}). "
                    "The on-disk cache changed since the dataset was created; "
                    "restart training so the cache and row selection are rebuilt."
                )
            self._table = table

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a single sample.

        Returns a dict matching the legacy format:
        {"image_id": str, "tags": List[str], "rating": str, "dir": Path}
        Plus "json_stem" if available (v2.0+ cache format).
        """
        # Lazy-load table on first access (enables parallel worker spawn)
        self._ensure_table()
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
        # Include exact image filename if available (caches built after the
        # filename column was added) — lets path resolution skip extension probing
        if "filename" in self._table.column_names:
            result["filename"] = row.column("filename")[0].as_py()
        return result

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare for pickling - serialize the cache path and row selection."""
        return {
            "_cache_path": self._cache_path,
            "_len": self._len,
            "_row_indices": self._row_indices,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore from pickle in worker process - defer mmap until first access.

        IMPORTANT: We do NOT load the Arrow table here. Loading during unpickle
        blocks each worker sequentially on Windows (spawn context), causing
        ~3.3s delay per worker (50+ seconds for 16 workers).

        Instead, we set _table = None and lazy-load on first __getitem__ call.
        This allows all workers to spawn in parallel, then each opens the
        memory-mapped file when it actually needs data.
        """
        self._cache_path = state["_cache_path"]
        self._len = state["_len"]
        # Row selection into the full on-disk cache (None = unfiltered).
        # .get() tolerates accessors pickled before this field existed.
        self._row_indices = state.get("_row_indices")
        self._table = None  # Lazy load on first access via _ensure_table()


class WorkerInitializer:
    """Picklable worker initialization callable for DataLoader.

    Handles RNG seeding and logging queue setup in worker processes.
    Unlike closures, class instances are picklable by default.

    Note: the vocabulary is small (~1-2 MB) and is pickled into workers with
    the dataset; the former shared-memory vocab path duplicated that work and
    was removed.
    """

    def __init__(self, log_queue=None, worker_log_level="WARNING"):
        """
        Args:
            log_queue: Queue for logging (optional)
            worker_log_level: Log level for workers (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                             WARNING allows debugging issues, CRITICAL minimizes queue overhead
        """
        self.log_queue = log_queue
        self.worker_log_level = worker_log_level

    def __call__(self, worker_id: int):
        """Worker initialization function called by DataLoader.

        Args:
            worker_id: Worker process ID
        """
        # Seed stdlib random and numpy from torch.initial_seed() so augmentations
        # (IndependentColorJitter, apply_random_rotation, blur) are reproducible
        # across resumes. PyTorch only auto-seeds torch.manual_seed in workers; the
        # stdlib `random` module and numpy stay unseeded by default, so any sampler
        # state we restore is undermined by drifting augmentation RNGs.
        try:
            base_seed = torch.initial_seed() % (2**32)
            random.seed(base_seed)
            try:
                import numpy as _np
                _np.random.seed(base_seed)
            except ImportError:
                pass
        except Exception:
            pass

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
                # Set worker log level from config (default: WARNING for debuggability)
                # CRITICAL minimizes queue overhead but hides worker errors
                # WARNING allows debugging issues with ~1-2ms/batch overhead
                level = getattr(logging, self.worker_log_level.upper(), logging.WARNING)
                logger.setLevel(level)
            except Exception:
                pass


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
        vocab=None,  # Optional TagVocabulary for num_classes validation
        # Image pipeline params
        image_size: int = 512,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        color_order: str = "RGB",
        preload_files: int = 0,
        # Dtype configuration
        tag_vector_dtype: str = "bfloat16",
    ):
        """
        Dataset loader for images and JSON metadata.
        Note: Despite legacy naming, this does NOT handle HDF5 files.
        """
        # Logger must exist before _load_annotations (which logs progress)
        self.logger = logging.getLogger(__name__)
        self.annotations = self._load_annotations(annotations_path)
        self.image_dir = image_dir
        self.transform = transform
        self.joint_transforms = joint_transforms
        self.max_retries = max_retries
        # Validate num_classes to prevent dimension mismatches in tag vectors
        if num_classes is not None and vocab is not None:
            vocab_size = len(vocab.tag_to_index)
            if num_classes != vocab_size:
                raise ValueError(
                    f"num_classes ({num_classes}) does not match vocabulary size ({vocab_size}). "
                    f"Pass num_classes=len(vocab.tag_to_index) for consistency."
                )
        elif vocab is not None and num_classes is None:
            # Auto-set num_classes from vocab size
            num_classes = len(vocab.tag_to_index)
            logging.info(f"DatasetLoader: auto-set num_classes={num_classes} from vocabulary size")
        elif num_classes is None:
            logging.warning(
                "DatasetLoader: num_classes not provided and no vocab available. "
                "Tag vectors may have incorrect dimensions. "
                "Pass num_classes=len(vocab.tag_to_index) for consistency."
            )
        self.num_classes = num_classes
        self.vocab = vocab
        # Use OrderedDict for O(1) FIFO eviction instead of O(n) list conversion
        self.retry_counts: OrderedDict[int, int] = OrderedDict()
        self.failed_samples = set()
        self._sample_error_log_count = 0  # Rate-limit error logs
        # Track error distribution per tag to detect bias
        # Memory bound: limited to _MAX_ERROR_STATS_TAGS unique tags (see _track_error_for_tags)
        # Once limit reached, new tags are not tracked but existing tags continue accumulating
        # Module-level factory (not a lambda) keeps this picklable for spawn workers
        self.error_stats = defaultdict(_nested_error_counter)
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
        # Counter to avoid checking exclusion staleness on every sample access
        # Only check every _EXCLUSION_CHECK_SAMPLE_INTERVAL samples (batch boundaries)
        self._exclusion_check_counter = 0

        # Image pipeline settings
        self.image_size = int(image_size)
        self.pad_color: Tuple[int, int, int] = (
            int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
        ) if isinstance(pad_color, (list, tuple)) else (114, 114, 114)
        self.normalize_mean: Tuple[float, float, float] = tuple(normalize_mean)
        self.normalize_std: Tuple[float, float, float] = tuple(normalize_std)
        _co = str(color_order or "RGB").upper()
        if _co not in ("RGB", "BGR"):
            raise ValueError(f"color_order must be 'RGB' or 'BGR', got {color_order!r}")
        self.color_order: str = _co

        # Tag vector dtype
        self._tag_vector_dtype = _canon_dtype(str(tag_vector_dtype).lower())

        # Epoch tracking for future flip support and consistency with SidecarJsonDataset.
        # Backed by shared memory so DataLoader workers (spawned at first iter()) observe
        # set_epoch() updates from the main process; a plain int would be frozen at fork/spawn time.
        self._current_epoch = mp.Value('i', 0, lock=False)

        # --- Pre-created transforms for performance (avoid recreating per sample) ---
        # Use v2 API to avoid deprecation warning (ToTensor is deprecated)
        # Image dtype for tensor outputs - bfloat16 for efficiency when supported
        self._image_dtype = torch.bfloat16
        if T is not None:
            self._to_tensor_v2 = T.Compose([T.ToImage(), T.ToDtype(self._image_dtype, scale=True)])
            self._to_tensor = self._to_tensor_v2  # Use v2 for v1 fallback too
        elif transforms is not None:
            self._to_tensor_v2 = None
            self._to_tensor = transforms.ToTensor()  # Legacy fallback
            self._image_dtype = torch.float32  # ToTensor outputs float32
        else:
            self._to_tensor_v2 = None
            self._to_tensor = None
            self._image_dtype = torch.float32  # Default fallback dtype

    # ---------- Pickling support for multiprocessing ----------
    def __getstate__(self):
        """Prepare for pickling - exclude unpicklable objects."""
        state = self.__dict__.copy()
        # Remove unpicklable objects before sending to worker
        state['_exclusion_manager'] = None   # Contains threading lock (will be recreated)
        # Snapshot exclusions for lock-free worker startup
        # Workers restore from snapshot instead of blocking on file lock
        state['_excluded_ids_snapshot'] = set(getattr(self, 'excluded_image_ids', set()))
        return state

    def __setstate__(self, state):
        """Restore from pickle in worker process."""
        self.__dict__.update(state)
        # Recreate exclusion manager in worker process (mirrors SidecarJsonDataset)
        exclusion_base = Path(self.dataset_root) if self.dataset_root else Path(self.image_dir).parent
        exclusion_path = exclusion_base / 'cache_exclusions.txt'
        self._exclusion_manager = ExclusionManager(
            exclusion_path,
            reload_interval_seconds=_EXCLUSION_RELOAD_INTERVAL
        )
        # Lock-free restore from snapshot (avoids file lock contention at startup)
        snapshot = state.get('_excluded_ids_snapshot', set())
        self.excluded_image_ids = snapshot
        self._exclusion_manager._excluded_ids = snapshot.copy()
        self._exclusion_manager._last_load_time = time.time()  # Treat snapshot as fresh
        # Set _last_mtime to current file mtime to prevent reload_if_stale()
        # from triggering _load_internal() on first __getitem__ call
        try:
            if self._exclusion_manager.exclusion_path.exists():
                self._exclusion_manager._last_mtime = self._exclusion_manager.exclusion_path.stat().st_mtime
        except OSError:
            pass  # File doesn't exist yet, that's fine

    def _get_image_path_for_idx(self, idx: int) -> Path:
        """Get image path for a given index.

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

        # Encode rating as a tag in the multi-hot vector
        rating_tag = _map_rating_to_tag(annotation.get("rating", "unknown"))
        if rating_tag and hasattr(self, 'vocab') and self.vocab is not None:
            rating_idx = self.vocab.tag_to_index.get(rating_tag)
            if rating_idx is not None and 0 <= rating_idx < tag_vec.shape[0]:
                tag_vec[rating_idx] = 1.0

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
            "image_id": image_id,
            "cached": cached,
            "error": False,
            "error_reason": "",
        }

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

        # Save to binary cache for faster future loads
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
        self._current_epoch.value = int(epoch)
        self.logger.debug(f"Dataset epoch set to {self._current_epoch.value}")

    def __getitem__(self, idx):
        # PERF: Only check exclusion staleness every N samples (batch boundaries)
        # instead of on every single sample access. This reduces function call
        # overhead from O(samples) to O(samples/N) where N=_EXCLUSION_CHECK_SAMPLE_INTERVAL
        self._exclusion_check_counter += 1
        if self._exclusion_manager and self._exclusion_check_counter >= _EXCLUSION_CHECK_SAMPLE_INTERVAL:
            self._exclusion_check_counter = 0
            if self._exclusion_manager.reload_if_stale():
                new_exclusions = self._exclusion_manager.get_excluded_ids()
                if len(new_exclusions) > len(self.excluded_image_ids):
                    self.excluded_image_ids = new_exclusions

        # HL002 Fix: Return error sample immediately on failure, don't bias distribution
        if idx in self.failed_samples:
            failed_image_id = None
            try:
                failed_image_id = self.annotations[idx].get('image_id')
            except Exception:
                pass
            return self._create_error_sample(idx, "Previously failed sample", image_id=failed_image_id)

        # Track retries with memory bounds to prevent unbounded growth
        # PERF: Using OrderedDict for O(1) FIFO eviction via popitem(last=False)
        # instead of O(n) list(dict.keys()) conversion
        if idx not in self.retry_counts:
            # Evict oldest entries if at capacity (simple FIFO eviction)
            # Remove 20% of entries to reduce eviction frequency and amortize cost
            if len(self.retry_counts) >= _MAX_RETRY_COUNTS:
                num_to_remove = _MAX_RETRY_COUNTS // 5
                for _ in range(num_to_remove):
                    self.retry_counts.popitem(last=False)  # O(1) removal of oldest
            self.retry_counts[idx] = 0

        annotation = None
        raw_image_id = None
        safe_image_id = None
        filename = None
        img_path = None
        try:
            annotation = self.annotations[idx]
            raw_image_id = annotation.get('image_id')
            filename = annotation.get('filename') or annotation.get('file_name')
            # Enforce allowlist and strip any sneaky path components
            safe_image_id = sanitize_identifier(str(raw_image_id))

            # Check if this sample was excluded by another worker (cross-worker sync)
            if safe_image_id and safe_image_id in self.excluded_image_ids:
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)
                return self._create_error_sample(idx, f"Excluded: {safe_image_id}", image_id=safe_image_id)

            # Drop samples whose targets would be all-(or near-all-)negative — the
            # train/val loops filter these via the `error` flag, so they never
            # reach the loss. Without this guard the model would be trained that
            # such images have zero of every tag (and zero of every rating).
            label_list = annotation.get('labels') or []
            if (not label_list) or _map_rating_to_tag(annotation.get('rating')) is None:
                reason = (
                    "empty label list" if not label_list
                    else f"missing/unknown rating ({annotation.get('rating')!r})"
                )
                return self._create_error_sample(idx, reason, image_id=safe_image_id)

            # --- Load + transform (confined path) ---
            # Use the sanitized image identifier we derived above.
            # Allow symlink targets to live under the dataset root (manifest symlinks → shard files)
            img_path = validate_image_path(
                Path(self.image_dir),
                safe_image_id,
                allowed_external_roots=([Path(self.dataset_root)] if self.dataset_root else None),
            )

            # Load image from disk
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

            # If provided, run joint v2 transforms to keep image & mask aligned
            if self.joint_transforms is not None and T is not None and tv_tensors is not None:
                img_tv = tv_tensors.Image(canvas)
                mask_tv = tv_tensors.Mask(pmask.to(torch.uint8))  # 1=PAD, 0=valid
                # v2 ops automatically use NEAREST for Mask; geometry stays in sync
                img_tv, mask_tv = self.joint_transforms(img_tv, mask_tv)
                # Pre-norm 0..1 tensor for L1; then normalize for model
                # _to_tensor_v2 already converts to bfloat16 via ToDtype
                img_01 = self._to_tensor_v2(img_tv)  # 0..1 bfloat16
                # BGR channel flip (CHW) immediately before normalization
                if self.color_order == "BGR":
                    img_01 = img_01.flip(0)
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
                        # _to_tensor already converts to bfloat16 (aliased to _to_tensor_v2)
                        img_01 = transformed if isinstance(transformed, torch.Tensor) else self._to_tensor(transformed)
                        if self.color_order == "BGR":
                            img_01 = img_01.flip(0)
                        t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)
                    except Exception as e:
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Transform failed, using fallback: {e}")
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img_01 = self._to_tensor(canvas)
                        if self.color_order == "BGR":
                            img_01 = img_01.flip(0)
                        t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)
                else:
                    if self._to_tensor is None:
                        raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                    img_01 = self._to_tensor(canvas)
                    if self.color_order == "BGR":
                        img_01 = img_01.flip(0)
                    t = _normalize_preserve_dtype(img_01, self.normalize_mean, self.normalize_std)

            # Reset retry count on success
            self.retry_counts[idx] = 0

            # Build sample using helper to avoid duplication
            return self._build_sample_dict(
                t, pmask, annotation, safe_image_id, cached=False
            )

        except Exception as e:
            self.retry_counts[idx] += 1
            self._sample_error_log_count += 1
            # Rate-limit warning logs: log first, then every 100th
            if self._sample_error_log_count == 1 or self._sample_error_log_count % 100 == 0:
                self.logger.warning(
                    "Failed to load sample idx=%s image_id=%s filename=%s path=%s error=%s (total errors: %s)",
                    idx,
                    safe_image_id or raw_image_id,
                    filename,
                    img_path,
                    e,
                    self._sample_error_log_count,
                )

            # Track error distribution to detect bias
            error_type = 'load_failed' if 'load' in str(e).lower() else 'decode_failed'
            self._track_error_distribution(idx, error_type)

            if self.retry_counts[idx] >= self.max_retries:
                # Add to failed set with memory bounds
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)

                # Persist failed sample to exclusion file immediately
                try:
                    failed_image_id = safe_image_id
                    if not failed_image_id and raw_image_id:
                        failed_image_id = sanitize_identifier(str(raw_image_id))
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
                self.logger.error(
                    "Sample idx=%s image_id=%s exceeded max retries, marking as failed",
                    idx,
                    safe_image_id or raw_image_id,
                )
                return self._create_error_sample(idx, str(e), image_id=safe_image_id or raw_image_id)

            # Return error sample instead of silently advancing to next index
            return self._create_error_sample(idx, f"Temporary failure: {e}", image_id=safe_image_id or raw_image_id)

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

    def _create_error_sample(self, idx, reason, image_id: Optional[str] = None):
        """Create a clearly marked error sample"""
        # Default to a common square size when transform is unknown
        sz = int(getattr(self, "image_size", 224) or 224)
        # Ensure num_classes is valid to prevent shape mismatches during batching
        if not self.num_classes or self.num_classes <= 0:
            raise ValueError(
                f"Cannot create error sample: num_classes={self.num_classes} is invalid. "
                "Pass num_classes=len(vocab.tag_to_index) when creating DatasetLoader."
            )
        # Use configured image dtype to match normal sample dtype for batch collation
        img_dtype = getattr(self, "_image_dtype", torch.float32)
        resolved_image_id = str(image_id).strip() if image_id else ""
        if not resolved_image_id:
            resolved_image_id = f"error_{idx}"
        # NOTE: key set must match _build_sample_dict exactly — default_collate
        # raises KeyError on heterogeneous dicts. Manifest mode has no flip
        # support, so no flip_applied/flip_mode keys on either path.
        return {
            "images": torch.zeros((3, sz, sz), dtype=img_dtype),
            "padding_mask": torch.ones((sz, sz), dtype=torch.bool),
            "tag_labels": torch.zeros(self.num_classes, dtype=self._tag_vector_dtype),
            "image_id": resolved_image_id,
            "cached": False,
            "error": True,
            "error_reason": reason,
        }

class AugmentationStats:
    """Placeholder class for augmentation statistics."""
    pass


def validate_dataset(*args, **kwargs):
    """Placeholder dataset validation function.

    NOT IMPLEMENTED: this returns immediately without inspecting any inputs or
    labels. It is only reachable when ``config.debug.validate_input_data`` is
    enabled; warn loudly so that flag does not give false assurance that inputs
    were validated.
    """
    logging.getLogger(__name__).warning(
        "validate_dataset() is a no-op placeholder — config.debug.validate_input_data "
        "is enabled but NO input/label validation is actually performed."
    )
    return {}


class SidecarJsonDataset(Dataset):
    """Dataset that reads per-image JSON sidecars in the same folder as images.

    Each JSON is expected to contain at least:
      - filename: image file name (e.g., "12345.jpg")
      - tags: space-delimited string or list of tags
      - rating: optional rating string or int (safe/general/questionable/explicit/unknown)

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
        color_order: str = "RGB",
        # --- Horizontal flipping ---
        random_flip_prob: float = 0.0,
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
        # Color jitter augmentation (applied before normalization)
        color_jitter_enabled: bool = False,
        color_jitter_brightness: float = 0.1,
        color_jitter_brightness_p: float = 0.15,
        color_jitter_contrast: float = 0.1,
        color_jitter_contrast_p: float = 0.15,
        color_jitter_saturation: float = 0.1,
        color_jitter_saturation_p: float = 0.15,
        # Random erasing augmentation (applied BEFORE normalization, after ColorJitter)
        random_erasing_enabled: bool = False,
        random_erasing_p: float = 0.25,
        random_erasing_scale_min: float = 0.02,
        random_erasing_scale_max: float = 0.20,
        random_erasing_ratio_min: float = 0.3,
        random_erasing_ratio_max: float = 3.3,
        # Random rotation augmentation (applied after letterboxing, image only)
        random_rotation_enabled: bool = False,
        random_rotation_p: float = 0.3,
        random_rotation_min_degrees: float = 5.0,
        random_rotation_max_degrees: float = 10.0,
        # Gaussian blur augmentation (applied to PIL before letterboxing)
        gaussian_blur_enabled: bool = False,
        gaussian_blur_p: float = 0.15,
        gaussian_blur_kernel_size: int = 3,
        gaussian_blur_sigma_min: float = 0.1,
        gaussian_blur_sigma_max: float = 1.5,
    ):
        self.root = Path(root_dir)
        self.json_files = list(json_files)
        self.vocab = vocab
        self.transform = transform
        self.joint_transforms = joint_transforms
        self.max_retries = max_retries
        # Use OrderedDict for O(1) FIFO eviction instead of O(n) list conversion
        self.retry_counts: OrderedDict[int, int] = OrderedDict()
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
        # Counter to avoid checking exclusion staleness on every sample access
        # Only check every _EXCLUSION_CHECK_SAMPLE_INTERVAL samples (batch boundaries)
        self._exclusion_check_counter = 0

        # Image pipeline settings
        self.image_size = int(image_size)
        self.pad_color: Tuple[int, int, int] = (
            int(pad_color[0]), int(pad_color[1]), int(pad_color[2])
        ) if isinstance(pad_color, (list, tuple)) else (114, 114, 114)
        self.normalize_mean: Tuple[float, float, float] = tuple(normalize_mean)
        self.normalize_std: Tuple[float, float, float] = tuple(normalize_std)
        _co = str(color_order or "RGB").upper()
        if _co not in ("RGB", "BGR"):
            raise ValueError(f"color_order must be 'RGB' or 'BGR', got {color_order!r}")
        self.color_order: str = _co

        # Image tensor dtype (matches _to_tensor output; downgraded below if the
        # torchvision v2 API is unavailable so error samples match real samples)
        self._image_dtype = torch.bfloat16

        # Tag vector dtype
        self._tag_vector_dtype = _canon_dtype(str(tag_vector_dtype).lower())

        # --- Horizontal flipping state ---
        self.random_flip_prob = float(random_flip_prob or 0.0)
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

        # Telemetry queue retained for compatibility (no orientation stats are pushed)
        self._stats_queue = stats_queue

        # Epoch tracking for flip variation across epochs.
        # Backed by shared memory so DataLoader workers (spawned at first iter() under
        # persistent_workers=True) observe set_epoch() updates from the main process;
        # a plain int would be frozen at spawn time and freeze flip decisions at epoch=0.
        self._current_epoch = mp.Value('i', 0, lock=False)
        self._epoch_was_set = False  # Track if set_epoch() was ever called
        self._epoch_warning_issued = False  # Avoid spamming warnings

        # --- Pre-created transforms for performance (avoid recreating per sample) ---
        # Use v2 API to avoid deprecation warning (ToTensor is deprecated)
        if T is not None:
            self._to_tensor_v2 = T.Compose([T.ToImage(), T.ToDtype(torch.bfloat16, scale=True)])
            self._to_tensor = self._to_tensor_v2  # Use v2 for v1 fallback too
        elif transforms is not None:
            self._to_tensor_v2 = None
            self._to_tensor = transforms.ToTensor()  # Legacy fallback
            self._image_dtype = torch.float32  # ToTensor outputs float32
        else:
            self._to_tensor_v2 = None
            self._to_tensor = None
            self._image_dtype = torch.float32  # Default fallback dtype

        # Augmentation transforms (applied fresh each sample, NOT cached)
        self._color_jitter = None
        self._random_erasing = None

        if T is not None:
            if color_jitter_enabled:
                self._color_jitter = IndependentColorJitter(
                    brightness=color_jitter_brightness,
                    brightness_p=color_jitter_brightness_p,
                    contrast=color_jitter_contrast,
                    contrast_p=color_jitter_contrast_p,
                    saturation=color_jitter_saturation,
                    saturation_p=color_jitter_saturation_p,
                )

            if random_erasing_enabled:
                # Use value=0.5 (mid-gray) which normalizes to 0.0 (center of [-1,1] range)
                # Note: value='random' was problematic as it samples from N(0,1) producing values outside [0,1]
                self._random_erasing = T.RandomErasing(
                    p=random_erasing_p,
                    scale=(random_erasing_scale_min, random_erasing_scale_max),
                    ratio=(random_erasing_ratio_min, random_erasing_ratio_max),
                    value=0.5,
                )

        # Random rotation augmentation
        self._rotation_enabled = bool(random_rotation_enabled)
        self._rotation_p = float(random_rotation_p)
        self._rotation_min_deg = float(random_rotation_min_degrees)
        self._rotation_max_deg = float(random_rotation_max_degrees)
        if self._rotation_enabled:
            self.logger.info(
                f"Random rotation enabled: p={self._rotation_p}, "
                f"angle=[{self._rotation_min_deg}, {self._rotation_max_deg}] degrees"
            )

        # Gaussian blur augmentation (DeiT III 3-Augment; Touvron et al. ECCV 2022)
        # Applied to PIL before letterboxing so padding regions remain at pad_color.
        self._blur_enabled = bool(gaussian_blur_enabled)
        self._blur_p = float(gaussian_blur_p)
        self._blur = None
        if self._blur_enabled and T is not None:
            ksize = int(gaussian_blur_kernel_size)
            if ksize % 2 == 0:
                ksize += 1  # torchvision requires odd kernel
            sigma_min = float(gaussian_blur_sigma_min)
            sigma_max = float(gaussian_blur_sigma_max)
            self._blur = T.GaussianBlur(kernel_size=ksize, sigma=(sigma_min, sigma_max))
            self.logger.info(
                f"Gaussian blur enabled: p={self._blur_p}, "
                f"kernel={ksize}, sigma=[{sigma_min}, {sigma_max}]"
            )

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

                # Build ONE boolean keep-mask over the FULL table combining
                # (a) split membership, (b) exclusions, and (c) known-bad rows.
                # The resulting row indices are handed to ArrowMetadataAccessor
                # so DataLoader workers — which reload the full combined cache
                # from disk (see ArrowMetadataAccessor.__getstate__) — can
                # reconstruct the exact same row selection. Filtering the table
                # without recording indices caused train/val split aliasing in
                # workers (val served a subset of train).
                full_len = len(arrow_table)
                keep_mask = None  # None = keep every row

                # (a) If using prebuilt table (contains ALL files), select this dataset's files
                if prebuilt_arrow_table is not None and "json_stem" in arrow_table.column_names:
                    # Build lookup set of combined keys from our json_files
                    # Using vectorized PyArrow filtering (10-100x faster than Python loop)
                    our_keys = {f"{str(jp.parent)}/{jp.stem}" for jp in self.json_files}
                    our_keys_array = pa.array(list(our_keys))

                    # Create combined key column in Arrow table for vectorized matching
                    dir_col = arrow_table.column("dir")
                    stem_col = arrow_table.column("json_stem")
                    combined_keys = pc.binary_join_element_wise(dir_col, stem_col, "/")

                    # Vectorized membership test (much faster than Python loop)
                    keep_mask = pc.is_in(combined_keys, value_set=our_keys_array)
                    self.logger.info(
                        f"Split membership mask: {pc.sum(keep_mask).as_py():,} of {full_len:,} "
                        "cache rows belong to this split (vectorized)"
                    )

                # (b) Exclusions (vectorized: O(n) Arrow ops vs O(n*m) Python)
                if self.excluded_image_ids:
                    self.logger.info(
                        f"Filtering {len(self.excluded_image_ids)} exclusions from Arrow cache..."
                    )
                    exclusion_array = pa.array(list(self.excluded_image_ids))
                    image_id_col = arrow_table.column("image_id")
                    is_excluded = pc.is_in(image_id_col, value_set=exclusion_array)
                    not_excluded = pc.invert(is_excluded)  # Keep items NOT in exclusion set
                    keep_mask = not_excluded if keep_mask is None else pc.and_(keep_mask, not_excluded)

                # (c) Known-bad rows: empty tag lists or missing/unknown ratings
                # only ever produce error samples (see the __getitem__ guard,
                # kept as fallback for the non-Arrow path) — drop them once
                # here instead of re-discovering them every epoch. The rating
                # set must mirror _map_rating_to_tag()'s string mapping.
                has_tags = pc.greater(pc.list_value_length(arrow_table.column("tags")), 0)
                known_ratings = pa.array(
                    ["g", "general", "safe", "sensitive", "q", "questionable", "e", "explicit"]
                )
                rating_ok = pc.is_in(
                    pc.utf8_lower(pc.utf8_trim_whitespace(arrow_table.column("rating"))),
                    value_set=known_ratings,
                )
                good_rows = pc.and_(has_tags, rating_ok)
                keep_mask = good_rows if keep_mask is None else pc.and_(keep_mask, good_rows)

                # Materialize the row selection. uint32 is plenty (<4.3B rows)
                # and keeps the per-worker pickle payload compact (~4 bytes/row).
                keep_mask = pc.fill_null(keep_mask, False)
                row_indices = pc.indices_nonzero(keep_mask).to_numpy().astype(np.uint32)
                if len(row_indices) != full_len:
                    arrow_table = arrow_table.take(row_indices)
                else:
                    row_indices = None  # Nothing filtered — workers can use the on-disk table as-is

                self.items = ArrowMetadataAccessor(
                    arrow_table, self._arrow_cache_path, row_indices=row_indices
                )
                self._using_arrow = True
                self.logger.info(
                    f"Loaded {len(self.items):,} of {full_len:,} Arrow cache rows "
                    "(split/exclusion/bad-row filters applied; selection is worker-safe)"
                )
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
                        "filename": Path(fname).name,  # Exact image filename (skips extension probing)
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
        state['_stats_queue'] = None         # multiprocessing.Queue (cannot be pickled on Windows spawn)
        state['_exclusion_manager'] = None   # Contains threading lock (will be recreated)
        # json_files (~5.6M Path objects) is only used in __init__; pickling it
        # into every spawn worker costs ~1 GB RAM + tens of seconds per worker
        state['json_files'] = []
        # Snapshot exclusions for lock-free worker startup
        # Workers restore from snapshot instead of blocking on file lock
        if hasattr(self, 'excluded_image_ids'):
            state['_excluded_ids_snapshot'] = set(self.excluded_image_ids)
        else:
            state['_excluded_ids_snapshot'] = set()
        # ArrowMetadataAccessor handles its own __getstate__/__setstate__
        # It only pickles the path, then re-opens the mmap in worker
        return state

    def __setstate__(self, state):
        """Restore from pickle in worker process."""
        self.__dict__.update(state)
        # These will be lazily recreated when needed:
        # - _stats_queue stays None in workers (telemetry only from main process)
        # - ArrowMetadataAccessor re-opens the memory-mapped file automatically

        # Recreate exclusion manager in worker process
        # This allows each worker to persist failed samples independently
        exclusion_path = self.root / 'cache_exclusions.txt'
        self._exclusion_manager = ExclusionManager(
            exclusion_path,
            reload_interval_seconds=_EXCLUSION_RELOAD_INTERVAL
        )
        # Lock-free restore from snapshot (avoids file lock contention at startup)
        # Workers spawn serially on Windows; blocking on file lock here causes
        # sequential initialization (RAM fills one worker at a time)
        snapshot = state.get('_excluded_ids_snapshot', set())
        self.excluded_image_ids = snapshot
        self._exclusion_manager._excluded_ids = snapshot.copy()
        self._exclusion_manager._last_load_time = time.time()  # Treat snapshot as fresh
        # CRITICAL: Set _last_mtime to current file mtime to prevent reload_if_stale()
        # from triggering _load_internal() on first __getitem__ call (causes lock contention)
        try:
            if self._exclusion_manager.exclusion_path.exists():
                self._exclusion_manager._last_mtime = self._exclusion_manager.exclusion_path.stat().st_mtime
        except OSError:
            pass  # File doesn't exist yet, that's fine
        # Periodic reload_if_stale() (every 30s) will catch new exclusions during training
        self._needs_initial_exclusion_refresh = False

    def __len__(self) -> int:
        return len(self.items)

    def _get_image_path_for_idx(self, idx: int) -> Path:
        """Get image path for a given index.

        Returns:
            Path to image file

        Raises:
            Exception on any error (caught by prefetcher)
        """
        ann = self.items[idx]
        image_id = ann["image_id"]
        img_root = ann.get("dir", self.root)
        return validate_image_path(Path(img_root), image_id, filename=ann.get("filename"))

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
        self._current_epoch.value = int(epoch)
        self._epoch_was_set = True
        self.logger.debug(f"Dataset epoch set to {self._current_epoch.value}")

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
        seed_bytes = f"{image_id}|epoch{self._current_epoch.value}".encode("utf-8")
        h = zlib.crc32(seed_bytes) & 0xFFFFFFFF  # Ensure unsigned 32-bit
        v = h / 0xFFFFFFFF  # [0,1]
        return v < float(self.random_flip_prob)

    def _decide_flip_mode(self, image_id: str, tags: List[str]) -> str:
        """
        Decide flipping policy: 'none' | 'random' | 'force'
        Respects flip list first; then applies the per-image deterministic coin.
        """
        if self.respect_flip_list:
            if image_id in self._never_flip_ids:
                return "none"
            if image_id in self._force_flip_ids:
                return "force"
        if self.random_flip_prob <= 0:
            return "none"
        return "random" if self._deterministic_coin(image_id) else "none"

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

        Returns:
            Sample dict for training
        """
        # Encode rating as a tag in the multi-hot vector
        rating_tag = _map_rating_to_tag(rating)
        if rating_tag:
            rating_tag_idx = self.vocab.tag_to_index.get(rating_tag)
            if rating_tag_idx is not None and 0 <= rating_tag_idx < tag_vec.shape[0]:
                tag_vec[rating_tag_idx] = 1.0

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
            "image_id": image_id,
            "cached": cached,
            "flip_applied": flip_applied,
            "flip_mode": flip_mode,
            "error": False,
            "error_reason": "",
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # PERF: Only check exclusion staleness every N samples (batch boundaries)
        # instead of on every single sample access. This reduces function call
        # overhead from O(samples) to O(samples/N) where N=_EXCLUSION_CHECK_SAMPLE_INTERVAL
        self._exclusion_check_counter += 1
        if self._exclusion_manager and self._exclusion_check_counter >= _EXCLUSION_CHECK_SAMPLE_INTERVAL:
            self._exclusion_check_counter = 0
            if self._exclusion_manager.reload_if_stale():
                # Update local reference if new exclusions were found
                new_exclusions = self._exclusion_manager.get_excluded_ids()
                if len(new_exclusions) > len(self.excluded_image_ids):
                    self.excluded_image_ids = new_exclusions

        # Materialize the metadata row ONCE per access and reuse it everywhere
        # below — each Arrow fetch decodes the full row (including the tags
        # list), so fetching for the exclusion check and again for the
        # annotation doubled per-sample metadata decode every epoch.
        ann = None
        image_id = None
        try:
            if 0 <= idx < len(self.items):
                ann = self.items[idx]
                image_id = ann.get("image_id")
        except Exception:
            ann = None

        if idx in self.failed_samples:
            return self._error_sample(idx, "Previously failed sample", image_id=image_id)

        # Check if this sample was excluded by another worker (cross-worker sync)
        if image_id and image_id in self.excluded_image_ids:
            # Mark as failed in memory too to speed up subsequent checks
            if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                self.failed_samples.add(idx)
            return self._error_sample(idx, f"Excluded by other worker: {image_id}", image_id=image_id)

        # Track retries with memory bounds to prevent unbounded growth
        # PERF: Using OrderedDict for O(1) FIFO eviction via popitem(last=False)
        # instead of O(n) list(dict.keys()) conversion
        if idx not in self.retry_counts:
            # Evict oldest entries if at capacity (simple FIFO eviction)
            # Remove 20% of entries to reduce eviction frequency and amortize cost
            if len(self.retry_counts) >= _MAX_RETRY_COUNTS:
                num_to_remove = _MAX_RETRY_COUNTS // 5
                for _ in range(num_to_remove):
                    self.retry_counts.popitem(last=False)  # O(1) removal of oldest
            self.retry_counts[idx] = 0

        img_root = None
        img_path = None
        try:
            # Reuse the row fetched above; a failed fetch routes through the
            # standard retry/error path.
            if ann is None:
                raise IndexError(f"Failed to materialize metadata row for idx={idx}")
            # Use original tags directly for read-only operations (avoid unnecessary copy)
            original_tags = ann["tags"]  # No copy - read-only reference

            # Filter samples that would inject false negatives into the loss:
            #   - Empty tag list → encode_tags() returns all-zero, training the model
            #     to predict "no tags" for an arbitrary image.
            #   - Missing/unknown rating → none of the four rating:* indices are set,
            #     teaching the model that no rating applies. With ASL gamma_neg high,
            #     this systematically biases toward "no rating".
            # Both classes of bad data are routed to the existing error-sample path,
            # which the train/val loops already filter via the `error` flag.
            if not original_tags or _map_rating_to_tag(ann.get("rating")) is None:
                reason = (
                    "empty tag list" if not original_tags
                    else f"missing/unknown rating ({ann.get('rating')!r})"
                )
                return self._error_sample(idx, reason, image_id=image_id)

            # Decide whether to flip; vocabulary has no orientation-sensitive tags,
            # so tags pass through unchanged.
            mode = self._decide_flip_mode(image_id, original_tags)
            flip_bit = mode != "none"
            tags_now = original_tags

            # Resolve image path first (needed for both cache lookup and loading)
            # Exact filename (when the cache provides it) skips per-extension probing
            img_root = ann.get("dir", self.root)
            img_path = validate_image_path(Path(img_root), image_id, filename=ann.get("filename"))

            # Load image from disk
            # Fully decode and correct EXIF while file is open
            with Image.open(img_path) as pil_img:
                pil_img.load()
                # Performance optimization: Only call exif_transpose when EXIF orientation
                # data is actually present. exif_transpose() parses EXIF on every call,
                # which is wasteful for images without orientation metadata.
                exif = pil_img.getexif()
                if exif and exif.get(0x0112):  # 0x0112 = EXIF Orientation tag
                    pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                    rgba = pil_img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, self.pad_color)
                    alpha = rgba.getchannel("A")
                    bg.paste(rgba, mask=alpha)
                    pil = bg
                else:
                    pil = pil_img.convert("RGB")

                # Apply color jitter to PIL image BEFORE letterboxing onto canvas.
                # This ensures jitter only affects actual image content, not padding regions.
                # (Padding is added by process_image_cpu and should remain at pad_color)
                if self._color_jitter is not None:
                    pil = self._color_jitter(pil)

                # Gaussian blur (PIL, pre-letterbox) — keeps padding regions sharp at pad_color
                if self._blur is not None and random.random() < self._blur_p:
                    pil = self._blur(pil)

                canvas, pmask = process_image_cpu(pil, self.image_size, self.pad_color)

                # Random rotation (image only, mask unchanged — fill matches pad_color)
                if self._rotation_enabled and random.random() < self._rotation_p:
                    canvas = apply_random_rotation(
                        canvas, self._rotation_min_deg, self._rotation_max_deg, self.pad_color
                    )

            # NOTE: Flip is applied AFTER joint_transforms to ensure correct ordering
            # (transforms operate on canonical unflipped images, flip is applied last)
            # Joint v2 transforms keep geometry aligned with mask when used
            if self.joint_transforms is not None and T is not None and tv_tensors is not None:
                img_tv = tv_tensors.Image(canvas)
                mask_tv = tv_tensors.Mask(pmask.to(torch.uint8))
                img_tv, mask_tv = self.joint_transforms(img_tv, mask_tv)
                # _to_tensor_v2 already converts to bfloat16 via ToDtype
                img = self._to_tensor_v2(img_tv)

                # Note: Color jitter is applied to PIL image BEFORE letterboxing (see above)
                # to avoid jittering the padding regions.

                # Random erasing BEFORE normalization (fills with 0.5 mid-gray,
                # which normalizes to 0.0 - the center of [-1,1] range)
                if self._random_erasing is not None:
                    img = self._random_erasing(img)

                # BGR channel flip (CHW) immediately before normalization
                if self.color_order == "BGR":
                    img = img.flip(0)
                img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)

                pmask = mask_tv.to(torch.bool)
            else:
                # Fallback: color-only transforms permitted
                if self.transform:
                    try:
                        transformed = self.transform(canvas)
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        # _to_tensor already converts to bfloat16 (aliased to _to_tensor_v2)
                        img = transformed if isinstance(transformed, torch.Tensor) else self._to_tensor(transformed)

                        # Note: Color jitter is applied to PIL image BEFORE letterboxing (see above)

                        # Random erasing BEFORE normalization (value='random' samples from N(0,1)
                        # which produces values up to ±4, incompatible with normalized [-1,1] range)
                        if self._random_erasing is not None:
                            img = self._random_erasing(img)

                        if self.color_order == "BGR":
                            img = img.flip(0)
                        img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)
                    except Exception as e:
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"Transform failed, using fallback: {e}")
                        if self._to_tensor is None:
                            raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                        img = self._to_tensor(canvas)

                        # Note: Color jitter is applied to PIL image BEFORE letterboxing (see above)

                        # Random erasing BEFORE normalization (value='random' samples from N(0,1)
                        # which produces values up to ±4, incompatible with normalized [-1,1] range)
                        if self._random_erasing is not None:
                            img = self._random_erasing(img)

                        if self.color_order == "BGR":
                            img = img.flip(0)
                        img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)
                else:
                    if self._to_tensor is None:
                        raise ImportError("torchvision is required for DatasetLoader transforms. Please install torchvision.")
                    img = self._to_tensor(canvas)

                    # Note: Color jitter is applied to PIL image BEFORE letterboxing (see above)

                    # Random erasing BEFORE normalization (value='random' samples from N(0,1)
                    # which produces values up to ±4, incompatible with normalized [-1,1] range)
                    if self._random_erasing is not None:
                        img = self._random_erasing(img)

                    if self.color_order == "BGR":
                        img = img.flip(0)
                    img = _normalize_preserve_dtype(img, self.normalize_mean, self.normalize_std)

            # Encode labels (tags already account for flipping)
            tag_vec = self.vocab.encode_tags(tags_now)  # (V,)

            # Apply horizontal flip after all other transforms complete
            # Flip is applied at tensor level after all other transforms complete
            if flip_bit:
                # Performance optimization: Removed .contiguous() calls after flip.
                # Modern PyTorch handles non-contiguous tensors efficiently, and the
                # subsequent operations don't require contiguous memory layout.
                img = torch.flip(img, dims=[2])  # Flip width dimension (CHW format)
                pmask = torch.flip(pmask, dims=[1])  # Flip width dimension (HW format)

            self.retry_counts[idx] = 0
            return self._build_sample_dict(
                img, pmask, tag_vec, ann.get("rating", "unknown"), image_id,
                cached=False,
                flip_applied=flip_bit,
                flip_mode=mode,
            )

        except Exception as e:
            self.retry_counts[idx] += 1
            self._sample_error_log_count += 1
            # Rate-limit warning logs: log first, then every 100th
            if self._sample_error_log_count == 1 or self._sample_error_log_count % 100 == 0:
                self.logger.warning(
                    "Failed to load sample idx=%s image_id=%s path=%s error=%s (total errors: %s)",
                    idx,
                    image_id,
                    img_path,
                    e,
                    self._sample_error_log_count,
                )
            if self.retry_counts[idx] >= self.max_retries:
                # Add to failed set with memory bounds
                if len(self.failed_samples) < _MAX_FAILED_SAMPLES:
                    self.failed_samples.add(idx)

                # Persist failed sample to exclusion file immediately
                # This ensures the sample is skipped in future epochs/runs
                try:
                    failed_image_id = image_id
                    if not failed_image_id and idx < len(self.items):
                        failed_image_id = self.items[idx].get("image_id")
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
                self.logger.error(
                    "Sample idx=%s image_id=%s exceeded max retries, marking as failed",
                    idx,
                    image_id,
                )
                return self._error_sample(idx, str(e), image_id=image_id)
            return self._error_sample(idx, f"Temporary failure: {e}", image_id=image_id)

    def _error_sample(self, idx: int, reason: str, image_id: Optional[str] = None) -> Dict[str, Any]:
        # Always use self.image_size for consistency with actual samples
        # This ensures error samples have the same shape as valid samples for batching
        sz = self.image_size  # Already int from __init__
        # Match image dtype to what cached/non-cached samples use to avoid batch collation issues
        # Match image dtype to what _to_tensor_v2 produces
        img_dtype = self._image_dtype
        resolved_image_id = str(image_id).strip() if image_id else ""
        if not resolved_image_id:
            resolved_image_id = f"error_{idx}"
        return {
            "images": torch.zeros((3, sz, sz), dtype=img_dtype),
            "padding_mask": torch.ones((sz, sz), dtype=torch.bool),
            "tag_labels": torch.zeros(len(self.vocab.tag_to_index), dtype=self._tag_vector_dtype),
            "image_id": resolved_image_id,
            "cached": False,
            "flip_applied": False,
            "flip_mode": "none",
            "error": True,
            "error_reason": reason,
        }


def _map_rating_to_tag(rating: Any) -> Optional[str]:
    """Map dataset rating field to a rating tag string.

    Returns the rating tag name (e.g., "rating:general") or None for unknown.
    Unknown ratings produce no rating tag, so the model learns nothing for
    that sample's rating (all rating tag positions stay 0 in the multi-hot vector).

    Args:
        rating: Rating value from dataset (int or str)

    Returns:
        Rating tag string, or None for unknown/invalid ratings
    """
    _IDX_TO_TAG = {
        0: "rating:general",
        1: "rating:sensitive",
        2: "rating:questionable",
        3: "rating:explicit",
    }

    if isinstance(rating, int):
        return _IDX_TO_TAG.get(int(rating))

    r = str(rating).strip().lower()
    _STR_TO_TAG = {
        "g": "rating:general", "general": "rating:general", "safe": "rating:general",
        "sensitive": "rating:sensitive",
        "q": "rating:questionable", "questionable": "rating:questionable",
        "e": "rating:explicit", "explicit": "rating:explicit",
    }
    return _STR_TO_TAG.get(r)


def create_dataloaders(
    data_config,
    validation_config,
    vocab_path,
    active_data_path,
    seed=42,
    debug_config=None,
    architecture_type: str = "vit",
    patch_size: Optional[int] = None,
    **kwargs,
):
    logger = logging.getLogger(__name__)

    # Extract config once to avoid redundant processing
    config_cache = {
        'preload_files': int(getattr(data_config, "preload_files", 0)),
        # Image processing configuration
        'image_size': int(getattr(data_config, "image_size", 512)),
        'normalize_mean': tuple(getattr(data_config, "normalize_mean", [0.5, 0.5, 0.5])),
        'normalize_std': tuple(getattr(data_config, "normalize_std", [0.5, 0.5, 0.5])),
        'pad_color': tuple(getattr(data_config, "pad_color", [114, 114, 114])),
        'color_order': str(getattr(data_config, "color_order", "RGB")).upper(),
        # Horizontal flip configuration
        'random_flip_prob': float(getattr(data_config, "random_flip_prob", 0.0)),
        'flip_overrides_path': getattr(data_config, "flip_overrides_path", None),
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
        # Color jitter augmentation
        'color_jitter_enabled': bool(getattr(data_config, "color_jitter_enabled", False)),
        'color_jitter_brightness': float(getattr(data_config, "color_jitter_brightness", 0.1)),
        'color_jitter_brightness_p': float(getattr(data_config, "color_jitter_brightness_p", 0.15)),
        'color_jitter_contrast': float(getattr(data_config, "color_jitter_contrast", 0.1)),
        'color_jitter_contrast_p': float(getattr(data_config, "color_jitter_contrast_p", 0.15)),
        'color_jitter_saturation': float(getattr(data_config, "color_jitter_saturation", 0.1)),
        'color_jitter_saturation_p': float(getattr(data_config, "color_jitter_saturation_p", 0.15)),
        # Random erasing augmentation
        'random_erasing_enabled': bool(getattr(data_config, "random_erasing_enabled", False)),
        'random_erasing_p': float(getattr(data_config, "random_erasing_p", 0.25)),
        'random_erasing_scale_min': float(getattr(data_config, "random_erasing_scale_min", 0.02)),
        'random_erasing_scale_max': float(getattr(data_config, "random_erasing_scale_max", 0.20)),
        'random_erasing_ratio_min': float(getattr(data_config, "random_erasing_ratio_min", 0.3)),
        'random_erasing_ratio_max': float(getattr(data_config, "random_erasing_ratio_max", 3.3)),
        # Random rotation augmentation
        'random_rotation_enabled': bool(getattr(data_config, "random_rotation_enabled", False)),
        'random_rotation_p': float(getattr(data_config, "random_rotation_p", 0.3)),
        'random_rotation_min_degrees': float(getattr(data_config, "random_rotation_min_degrees", 5.0)),
        'random_rotation_max_degrees': float(getattr(data_config, "random_rotation_max_degrees", 10.0)),
        # Gaussian blur augmentation
        'gaussian_blur_enabled': bool(getattr(data_config, "gaussian_blur_enabled", False)),
        'gaussian_blur_p': float(getattr(data_config, "gaussian_blur_p", 0.15)),
        'gaussian_blur_kernel_size': int(getattr(data_config, "gaussian_blur_kernel_size", 3)),
        'gaussian_blur_sigma_min': float(getattr(data_config, "gaussian_blur_sigma_min", 0.1)),
        'gaussian_blur_sigma_max': float(getattr(data_config, "gaussian_blur_sigma_max", 1.5)),
    }

    # Load vocabulary once (needed for sidecar mode and to determine num classes)
    # The vocab is small (~1-2 MB) and is simply pickled into workers with the
    # dataset — the former shared-memory vocab path duplicated that work.
    vocab = load_vocabulary_for_training(Path(vocab_path))
    num_tags = len(vocab.tag_to_index)

    image_size = config_cache['image_size']
    pad_color = config_cache['pad_color']
    transform = None

    # ViT uses inception-style normalization (typically 0.5/0.5/0.5 mean/std).
    # Stats are taken AS-IS from config; they are interpreted in the same
    # channel order as the image (see data.color_order).
    mean = config_cache['normalize_mean']
    std = config_cache['normalize_std']
    logger.info(
        f"Using ViT (inception-style) normalization: mean={mean}, std={std} "
        f"(color_order={config_cache['color_order']})"
    )

    # Determine dataset mode
    root = Path(active_data_path)
    manifest_train = root / "train.json"
    manifest_val = root / "val.json"
    images_dir = root / "images"

    # Flip configuration
    random_flip_prob = config_cache['random_flip_prob']
    flip_overrides_path = config_cache['flip_overrides_path']

    if manifest_train.exists() and manifest_val.exists() and images_dir.exists():
        # Manifest mode (back-compat); legacy DatasetLoader does not support flips.
        if float(getattr(data_config, "random_flip_prob", 0.0) or 0.0) > 0.0:
            logger.warning(
                "random_flip_prob > 0 with manifest dataset; legacy DatasetLoader does not "
                "support flips, disabling."
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
            color_order=config_cache['color_order'],
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
            color_order=config_cache['color_order'],
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
            color_order=config_cache['color_order'],
            random_flip_prob=random_flip_prob,
            flip_overrides_path=flip_overrides_path,
            stats_queue=config_cache['stats_queue'],
            metadata_cache_enabled=config_cache['metadata_cache_enabled'],
            metadata_cache_workers=config_cache['metadata_cache_workers'],
            force_rebuild_metadata_cache=False,  # Already built above
            metadata_cache_staleness_check_samples=config_cache['metadata_cache_staleness_check_samples'],
            prebuilt_arrow_table=prebuilt_arrow_table,
            # Augmentation (training only)
            color_jitter_enabled=config_cache['color_jitter_enabled'],
            color_jitter_brightness=config_cache['color_jitter_brightness'],
            color_jitter_brightness_p=config_cache['color_jitter_brightness_p'],
            color_jitter_contrast=config_cache['color_jitter_contrast'],
            color_jitter_contrast_p=config_cache['color_jitter_contrast_p'],
            color_jitter_saturation=config_cache['color_jitter_saturation'],
            color_jitter_saturation_p=config_cache['color_jitter_saturation_p'],
            random_erasing_enabled=config_cache['random_erasing_enabled'],
            random_erasing_p=config_cache['random_erasing_p'],
            random_erasing_scale_min=config_cache['random_erasing_scale_min'],
            random_erasing_scale_max=config_cache['random_erasing_scale_max'],
            random_erasing_ratio_min=config_cache['random_erasing_ratio_min'],
            random_erasing_ratio_max=config_cache['random_erasing_ratio_max'],
            random_rotation_enabled=config_cache['random_rotation_enabled'],
            random_rotation_p=config_cache['random_rotation_p'],
            random_rotation_min_degrees=config_cache['random_rotation_min_degrees'],
            random_rotation_max_degrees=config_cache['random_rotation_max_degrees'],
            gaussian_blur_enabled=config_cache['gaussian_blur_enabled'],
            gaussian_blur_p=config_cache['gaussian_blur_p'],
            gaussian_blur_kernel_size=config_cache['gaussian_blur_kernel_size'],
            gaussian_blur_sigma_min=config_cache['gaussian_blur_sigma_min'],
            gaussian_blur_sigma_max=config_cache['gaussian_blur_sigma_max'],
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
            color_order=config_cache['color_order'],
            random_flip_prob=0.0,          # keep val deterministic
            flip_overrides_path=None,
            stats_queue=config_cache['stats_queue'],
            # Now sharing prebuilt cache properly - no file count mismatch
            metadata_cache_enabled=config_cache['metadata_cache_enabled'],
            metadata_cache_workers=config_cache['metadata_cache_workers'],
            force_rebuild_metadata_cache=False,  # Already built above
            metadata_cache_staleness_check_samples=config_cache['metadata_cache_staleness_check_samples'],
            prebuilt_arrow_table=prebuilt_arrow_table,
            # No augmentation for validation (deterministic evaluation)
            color_jitter_enabled=False,
            random_erasing_enabled=False,
            random_rotation_enabled=False,
        )

    # ResumableSampler extends DistributedSampler with O(1) mid-epoch resume support
    # by allowing direct offset into the shuffled indices instead of iterating through.
    # Uses num_replicas=1, rank=0 for single-GPU training.
    train_sampler = ResumableSampler(
        train_ds,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=config_cache['drop_last'],
        seed=int(seed) if seed is not None else 0,
    )

    val_sampler = None
    # --------------------------------------------------------------------

    # DataLoaders
    def _dl_kwargs(cfg, *, shuffle: bool, drop_last: bool, override_cfg=None):
        """Build DataLoader kwargs from config.

        Args:
            cfg: Primary config (data_config)
            shuffle: Whether to shuffle
            drop_last: Whether to drop last incomplete batch
            override_cfg: Optional override config (e.g., validation_config.dataloader)
                         Values from override_cfg take precedence over cfg.
        """
        # Use override values if provided, else fall back to primary config
        def get_val(attr, default=None):
            if override_cfg is not None and hasattr(override_cfg, attr):
                return getattr(override_cfg, attr)
            return getattr(cfg, attr, default)

        kw = dict(
            batch_size=get_val("batch_size"),
            num_workers=get_val("num_workers"),
            pin_memory=get_val("pin_memory", True),
            drop_last=drop_last,
            shuffle=shuffle,
        )
        # Only use multiprocessing knobs when workers > 0
        num_workers = int(kw.get("num_workers", 0) or 0)
        if num_workers > 0:
            prefetch = get_val("prefetch_factor", None)
            if prefetch is not None:
                kw["prefetch_factor"] = prefetch
            kw["persistent_workers"] = bool(get_val("persistent_workers", False))
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
    worker_log_level = getattr(data_config, "worker_log_level", "WARNING")
    _train_kw["worker_init_fn"] = WorkerInitializer(log_queue, worker_log_level)
    train_loader = DataLoader(train_ds, **_train_kw)

    # Build validation loader kwargs using validation-specific config if available
    val_override_cfg = validation_config.dataloader if hasattr(validation_config, "dataloader") else None
    _val_kw = _dl_kwargs(data_config, shuffle=False, drop_last=False, override_cfg=val_override_cfg)
    if val_sampler is not None:
        _val_kw["sampler"] = val_sampler
    _val_kw["worker_init_fn"] = WorkerInitializer(log_queue, worker_log_level)
    val_loader = DataLoader(val_ds, **_val_kw)

    # Log validation dataloader settings for visibility
    if val_override_cfg is not None:
        logger.info(
            f"Validation DataLoader: batch_size={_val_kw['batch_size']}, "
            f"num_workers={_val_kw['num_workers']}, "
            f"prefetch_factor={_val_kw.get('prefetch_factor', 'default')}"
        )

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
