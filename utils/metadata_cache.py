"""
Metadata cache for SidecarJsonDataset.

Caches parsed JSON metadata (image_id, tags, rating, dir) to avoid
re-parsing millions of JSON files on every training run.

Uses parallel processing with ThreadPoolExecutor to speed up initial cache
creation, and safetensors for safe, fast serialization.
"""

from __future__ import annotations
import gc
import hashlib
import io
import json
import logging
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Try to use orjson for faster JSON parsing (5-10x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# Import required functions from utils
from utils.path_utils import sanitize_identifier
from utils.metadata_ingestion import parse_tags_field

# Import safetensors for serialization
from safetensors.torch import save_file, load_file
import torch

# File locking for concurrent access prevention (required for data integrity)
try:
    import filelock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
    import warnings
    warnings.warn(
        "filelock not installed - metadata cache writes are NOT thread-safe. "
        "Install with: pip install filelock",
        RuntimeWarning
    )


_PROJ_ROOT = Path(__file__).resolve().parent.parent
_CACHE_VERSION = "2.0"

# Staleness check timeout (seconds) - configurable via env var for NAS/slow filesystems
_STALENESS_TIMEOUT = int(os.environ.get("CACHE_STALENESS_TIMEOUT", "120"))


def _compute_file_list_hash(json_files: List[Path]) -> str:
    """Compute stable hash of json file list for cache validation.

    For large datasets (>3000 files), samples first 1000, middle 1000, and last 1000
    files for performance while maintaining good coverage.
    """
    # Sort to ensure consistency
    sorted_paths = sorted(str(p.resolve()) for p in json_files)

    # Sample if too large (hash first 1000, last 1000, middle 1000)
    if len(sorted_paths) > 3000:
        mid = len(sorted_paths) // 2
        sample = sorted_paths[:1000] + sorted_paths[mid-500:mid+500] + sorted_paths[-1000:]
    else:
        sample = sorted_paths

    combined = "\n".join(sample)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _compute_sample_size(total_files: int, requested_samples: int) -> int:
    """Compute adaptive sample size based on dataset size.

    Scales from 100% for small datasets to 0.04% for huge datasets (5M+).
    Reduced from 10K to 2K max for huge datasets to speed up staleness checks.
    """
    if total_files < 1000:
        return total_files  # Check all files for small datasets
    elif total_files < 100000:
        return min(requested_samples, total_files // 10)  # 10% for medium datasets
    elif total_files < 1000000:
        return min(requested_samples * 5, 3000)  # 500-3000 for large datasets
    else:
        return min(requested_samples * 10, 2000)  # 1000-2000 for huge datasets (0.04% of 5M)


def _stratified_sample(json_files: List[Path], sample_size: int) -> List[Path]:
    """Sample files from different directories for better coverage."""
    if sample_size >= len(json_files):
        return json_files

    # Group by parent directory
    by_dir = {}
    for f in json_files:
        parent = f.parent
        if parent not in by_dir:
            by_dir[parent] = []
        by_dir[parent].append(f)

    # Sample proportionally from each directory
    samples = []
    dirs = list(by_dir.keys())
    samples_per_dir = max(1, sample_size // len(dirs))
    remainder = sample_size % len(dirs)

    for i, dir_path in enumerate(dirs):
        dir_sample_size = samples_per_dir + (1 if i < remainder else 0)
        dir_files = by_dir[dir_path]
        samples.extend(random.sample(dir_files, min(dir_sample_size, len(dir_files))))

    # Ensure we don't exceed sample_size due to rounding
    return samples[:sample_size]


def _metadata_cache_path(root: Path) -> Path:
    """Return cache file path for metadata cache for a given dataset root.

    Uses same hashing as split cache for consistency.
    Files live under ./logs/metadata_cache/<sha1(root)>.metadata.safetensors
    """
    cache_dir = _PROJ_ROOT / "logs" / "metadata_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}.metadata.safetensors"


def _validate_cache(
    cache_data: Dict[str, Any],
    expected_count: int,
    json_files: Optional[List[Path]],
    logger: logging.Logger
) -> bool:
    """Validate loaded cache data for integrity (v2.0 with file list validation).

    Args:
        cache_data: Loaded cache dictionary
        expected_count: Expected number of items
        json_files: Optional list of JSON files for hash validation (None for backward compat)
        logger: Logger instance

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check version field exists
        if "version" not in cache_data:
            logger.warning("Metadata cache missing version field. Cache will be rebuilt.")
            return False

        version = cache_data["version"]
        # Decode version from tensor/array of ASCII bytes (robust handling)
        if hasattr(version, 'tolist'):
            # Works for both torch.Tensor and numpy arrays
            version_bytes = version.tolist()
        elif isinstance(version, (list, tuple)):
            version_bytes = list(version)
        else:
            version_bytes = None

        if version_bytes and all(isinstance(x, (int, float)) for x in version_bytes):
            # Decode ASCII bytes to string
            version = "".join(chr(int(x)) for x in version_bytes)
        else:
            version = str(version)

        if version != _CACHE_VERSION:
            logger.info(
                f"Metadata cache version mismatch: {version} != {_CACHE_VERSION}. "
                "Rebuilding with current version..."
            )
            return False

        # Check required fields exist
        required_fields = ["image_ids", "tags", "ratings", "dirs", "created_at"]
        for field in required_fields:
            if field not in cache_data:
                logger.warning(f"Metadata cache missing required field: {field}. Cache will be rebuilt.")
                return False

        # Validate file list if json_files provided (v2.0 feature)
        if json_files is not None:
            # NOTE: Skipping file_list_hash validation - it's too expensive for large datasets
            # (recomputing hash requires resolving all 5M+ paths which blocks for minutes).
            # The file count check below provides sufficient validation.
            # The file_list_hash field is still stored in the cache for future use.

            # Validate file count
            if "file_list_count" in cache_data:
                cached_count_tensor = cache_data["file_list_count"]
                cached_count = int(cached_count_tensor.item() if isinstance(cached_count_tensor, torch.Tensor) else cached_count_tensor)
                actual_count = len(json_files)

                # Tighter tolerance: 0.1% (down from 1%)
                tolerance = max(100, int(actual_count * 0.001))
                if abs(cached_count - actual_count) > tolerance:
                    logger.info(
                        f"File count changed: cache={cached_count}, current={actual_count}, "
                        f"diff={abs(cached_count - actual_count)}, tolerance={tolerance} (0.1%). "
                        "Rebuilding metadata cache..."
                    )
                    return False

        # Check stored count field (v2.0: replaces incorrect byte-length comparison)
        # NOTE: Previously compared tensor.size(0) which returns BYTE count, not entry count.
        # Each field is stored as raw bytes (uint8), so byte lengths differ by field type.
        if "count" not in cache_data:
            logger.warning("Cache missing 'count' field. Cache will be rebuilt.")
            return False

        count_tensor = cache_data["count"]
        cached_count = int(count_tensor.item() if isinstance(count_tensor, torch.Tensor) else count_tensor)

        # Check count matches expectation (v2.0: tighter 0.1% tolerance, down from 1%)
        tolerance_threshold = max(100, int(expected_count * 0.001))
        if abs(cached_count - expected_count) > tolerance_threshold:
            logger.info(
                f"Metadata cache count mismatch: cached={cached_count}, expected={expected_count}, "
                f"diff={abs(cached_count - expected_count)}, tolerance={tolerance_threshold} (0.1%). "
                "Cache will be rebuilt."
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Metadata cache validation error: {e}. Cache will be rebuilt.")
        return False


def _is_cache_stale(
    cache_path: Path,
    json_files: List[Path],
    sample_size: int,
    logger: logging.Logger
) -> bool:
    """Check if cache is stale using improved v2.0 sampling approach.

    Uses:
    - Dynamic sample size (scales with dataset size)
    - Stratified sampling (directory-aware for better coverage)
    - Parallel checking (16 workers for faster stat calls)

    Args:
        cache_path: Path to cache file
        json_files: List of JSON file paths
        sample_size: Base sample size (will be scaled dynamically)
        logger: Logger instance

    Returns:
        True if cache is stale, False otherwise
    """
    try:
        # Get cache creation time
        cache_mtime = cache_path.stat().st_mtime

        # Compute dynamic sample size based on dataset size
        total_files = len(json_files)
        if total_files == 0:
            return False

        effective_sample_size = _compute_sample_size(total_files, sample_size)

        # Use stratified sampling for better coverage
        sample_files = _stratified_sample(json_files, effective_sample_size)

        logger.debug(
            f"Staleness check: sampling {len(sample_files)} files "
            f"({len(sample_files)/total_files*100:.2f}%) from {total_files} total"
        )

        # Parallel staleness checking with ThreadPoolExecutor
        stale_files = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            def check_file(json_file: Path) -> Optional[Path]:
                """Return file path if stale, None otherwise."""
                try:
                    if json_file.stat().st_mtime > cache_mtime:
                        return json_file
                except (OSError, FileNotFoundError):
                    # File no longer exists - might indicate staleness
                    pass
                return None

            futures = [executor.submit(check_file, f) for f in sample_files]
            # Use configurable timeout (default 120s) for NAS/slow filesystem compatibility
            for future in as_completed(futures, timeout=_STALENESS_TIMEOUT):
                try:
                    result = future.result()
                    if result is not None:
                        stale_files.append(result)
                        # Early exit on first stale file found
                        if len(stale_files) >= 1:
                            break
                except TimeoutError:
                    logger.warning("Staleness check timed out")
                    return True
                except Exception:
                    continue

        if stale_files:
            logger.info(
                f"Metadata cache stale: {stale_files[0]} modified after cache creation "
                f"(found {len(stale_files)} stale in sample)"
            )
            return True

        return False

    except Exception as e:
        logger.warning(f"Staleness check failed: {e}")
        # On error, assume cache is stale to be safe
        return True


def _parse_json_batch(
    json_files: List[Path],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Parse a batch of JSON files (worker function).

    Args:
        json_files: List of JSON file paths to parse
        logger: Logger instance

    Returns:
        List of parsed metadata dicts
    """
    # Use orjson if available (5-10x faster than stdlib json)
    if HAS_ORJSON:
        def load_json(path: Path):
            return orjson.loads(path.read_bytes())
    else:
        def load_json(path: Path):
            return json.loads(path.read_text(encoding="utf-8"))

    items = []
    for jp in json_files:
        try:
            data = load_json(jp)
            # Skip if data is not a dict (e.g., manifest files are lists)
            if not isinstance(data, dict):
                logger.warning(f"Skipping {jp}: expected dict, got {type(data).__name__}")
                continue
            fname = str(data.get("filename") or jp.with_suffix(".png").name)
            image_id = sanitize_identifier(Path(fname).stem)
            tags_raw = data.get("tags")
            tags_list = parse_tags_field(tags_raw)
            rating = data.get("rating", "unknown")

            items.append({
                "image_id": image_id,
                "tags": tags_list,
                "rating": rating,
                "dir": str(jp.parent),  # Store as string for serialization
            })
        except Exception as e:
            # Log warning but continue processing
            logger.warning(f"Failed to parse {jp}: {e}")
            continue

    return items


def _encode_to_bytes(
    items: List[Dict[str, Any]],
    key: str,
    transform=str,
    logger: Optional[logging.Logger] = None,
    field_name: str = ""
) -> bytearray:
    """Incrementally encode items to bytes without huge string allocations.

    Uses io.BytesIO for efficient incremental encoding instead of joining
    millions of strings into a massive intermediate string.

    Args:
        items: List of metadata dicts
        key: Key to extract from each item
        transform: Function to convert value to string (default: str)
        logger: Optional logger for progress reporting
        field_name: Name of field being encoded (for progress messages)

    Returns:
        bytearray containing newline-separated encoded values
    """
    buffer = io.BytesIO()
    total = len(items)
    progress_interval = max(500000, total // 10)  # Log every 10% or 500K items
    start_time = time.time()

    for i, item in enumerate(items):
        if i > 0:
            buffer.write(b"\n")
        # Use errors='surrogateescape' to handle invalid UTF-8 sequences gracefully
        # This prevents encoding failures on malformed metadata while preserving data
        buffer.write(transform(item[key]).encode("utf-8", errors="surrogateescape"))

        # Progress logging
        if logger and (i + 1) % progress_interval == 0:
            pct = (i + 1) * 100 // total
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  Encoding {field_name}: {i + 1:,}/{total:,} ({pct}%) @ {rate:,.0f}/s")

    return bytearray(buffer.getvalue())


def _build_metadata_cache(
    cache_path: Path,
    json_files: List[Path],
    num_workers: int,
    logger: logging.Logger
) -> Optional[List[Dict[str, Any]]]:
    """Build metadata cache using parallel processing.

    Args:
        cache_path: Path where cache will be saved
        json_files: List of JSON file paths to parse
        num_workers: Number of parallel workers
        logger: Logger instance

    Returns:
        List of parsed metadata dicts, or None on failure
    """
    try:
        logger.info(
            f"Building metadata cache with {num_workers} workers "
            f"({len(json_files)} files)..."
        )
        start_time = time.time()

        # Split json_files into chunks for parallel processing
        # Larger chunks (4096) reduce thread coordination overhead for huge datasets
        # Target ~200-500 chunks for optimal parallelism
        total_files = len(json_files)
        if total_files > 1_000_000:
            chunk_size = 8192  # 8K chunks for 5M+ files (~700 chunks)
        elif total_files > 100_000:
            chunk_size = 4096  # 4K chunks for 100K-1M files
        else:
            chunk_size = 1024  # Original size for smaller datasets
        chunks = [
            json_files[i:i + chunk_size]
            for i in range(0, len(json_files), chunk_size)
        ]

        # Process chunks in parallel
        all_items = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(_parse_json_batch, chunk, logger): chunk
                for chunk in chunks
            }

            # Collect results with progress bar
            if HAS_TQDM:
                progress = tqdm(
                    total=len(json_files),
                    desc="Parsing metadata",
                    unit="files",
                    unit_scale=True
                )
            else:
                progress = None

            for future in as_completed(futures):
                try:
                    batch_items = future.result()
                    all_items.extend(batch_items)

                    if progress:
                        progress.update(len(futures[future]))
                    elif len(all_items) % 10000 == 0:
                        logger.info(
                            f"Parsed {len(all_items):,} / {len(json_files):,} files"
                        )
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
                    continue

            if progress:
                progress.close()

        elapsed = time.time() - start_time
        logger.info(
            f"Parsed {len(all_items):,} files in {elapsed:.1f}s "
            f"({len(all_items)/elapsed:.0f} files/sec)"
        )

        # Convert to columnar format for safetensors
        logger.info("Serializing metadata cache...")

        # Build byte buffers incrementally (avoids huge string allocations)
        # Using _encode_to_bytes instead of "\n".join() reduces memory pressure
        # and speeds up serialization for millions of items
        image_ids_bytes = _encode_to_bytes(
            all_items, "image_id", logger=logger, field_name="image_ids"
        )
        # Use orjson for serialization if available (faster than json.dumps)
        if HAS_ORJSON:
            def tags_transform(tags):
                return orjson.dumps(tags).decode("utf-8")
        else:
            tags_transform = json.dumps
        tags_bytes = _encode_to_bytes(
            all_items, "tags", transform=tags_transform, logger=logger, field_name="tags"
        )
        ratings_bytes = _encode_to_bytes(
            all_items, "rating", logger=logger, field_name="ratings"
        )
        dirs_bytes = _encode_to_bytes(
            all_items, "dir", logger=logger, field_name="dirs"
        )
        logger.info("  Writing cache file...")

        # Compute file list hash for v2.0 validation
        file_list_hash = _compute_file_list_hash(json_files)
        logger.debug(f"Metadata cache file list hash: {file_list_hash}")

        # Create tensors from byte buffers with explicit clone to prevent dangling refs
        # Using clone() ensures data is copied before bytearrays can be garbage collected
        cache_tensors = {
            "image_ids": torch.frombuffer(image_ids_bytes, dtype=torch.uint8).clone(),
            "tags": torch.frombuffer(tags_bytes, dtype=torch.uint8).clone(),
            "ratings": torch.frombuffer(ratings_bytes, dtype=torch.uint8).clone(),
            "dirs": torch.frombuffer(dirs_bytes, dtype=torch.uint8).clone(),
            "version": torch.tensor([ord(c) for c in _CACHE_VERSION], dtype=torch.uint8),
            "created_at": torch.tensor([time.time()], dtype=torch.float64),
            "count": torch.tensor([len(all_items)], dtype=torch.int64),
            # v2.0 fields for validation
            "file_list_hash": torch.tensor([ord(c) for c in file_list_hash], dtype=torch.uint8),
            "file_list_count": torch.tensor([len(json_files)], dtype=torch.int64),
        }

        # Calculate expected minimum size for post-write verification
        expected_data_size = (
            len(image_ids_bytes) + len(tags_bytes) + len(ratings_bytes) + len(dirs_bytes)
        )

        # Atomic write with optional file locking for concurrent access prevention
        temp_path = cache_path.with_suffix(".tmp")
        lock_path = cache_path.with_suffix(".lock")

        def _do_atomic_write():
            """Perform the atomic write operations."""
            save_file(cache_tensors, str(temp_path))
            # Atomic rename using os.replace() - works on all platforms and
            # automatically overwrites destination without TOCTOU race
            os.replace(str(temp_path), str(cache_path))

        # Use file locking if available (prevents concurrent write corruption)
        if HAS_FILELOCK:
            with filelock.FileLock(lock_path, timeout=600):  # 10 min timeout for large caches
                _do_atomic_write()
        else:
            _do_atomic_write()

        # Post-write verification: ensure file is complete and valid
        # Use safetensors safe_open to verify file integrity, not just size
        from safetensors import safe_open
        try:
            with safe_open(str(cache_path), framework="pt") as f:
                # Verify all expected keys are present
                saved_keys = set(f.keys())
                expected_keys = {"image_ids", "tags", "ratings", "dirs", "version", "created_at", "count"}
                missing_keys = expected_keys - saved_keys
                if missing_keys:
                    raise ValueError(f"Cache file missing keys after write: {missing_keys}")
        except Exception as e:
            raise ValueError(f"Cache file verification failed: {e}") from e

        saved_size = cache_path.stat().st_size
        if saved_size < expected_data_size * 0.3:  # Safetensors has some compression
            raise ValueError(
                f"Cache file too small after write: {saved_size} bytes "
                f"(expected at least {expected_data_size * 0.3:.0f} bytes)"
            )

        cache_size_mb = saved_size / (1024 * 1024)
        logger.info(
            f"Metadata cache saved to {cache_path} ({cache_size_mb:.1f} MB)"
        )

        return all_items

    except Exception as e:
        logger.error(f"Failed to build metadata cache: {e}", exc_info=True)
        # Clean up partial cache file
        try:
            if cache_path.exists():
                cache_path.unlink()
            temp_path = cache_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return None


def _load_cached_metadata(
    cache_path: Path,
    expected_count: int,
    json_files: Optional[List[Path]],
    logger: logging.Logger
) -> Optional[List[Dict[str, Any]]]:
    """Load metadata from cache file with v2.0 validation.

    Args:
        cache_path: Path to cache file
        expected_count: Expected number of items
        json_files: Optional list of JSON files for validation (v2.0 feature)
        logger: Logger instance

    Returns:
        List of parsed metadata dicts, or None on failure
    """
    try:
        logger.info(f"Loading metadata cache from {cache_path}...")
        start_time = time.time()

        # Load cache file
        cache_data = load_file(str(cache_path))
        load_elapsed = time.time() - start_time
        logger.info(f"  Cache file loaded in {load_elapsed:.1f}s, validating...")

        # Validate cache (v2.0: includes file list validation)
        if not _validate_cache(cache_data, expected_count, json_files, logger):
            return None

        # Pre-validate tensor sizes before expensive decode operation
        # This catches corruption early without wasting memory on decoding
        tensor_sizes = {
            "image_ids": cache_data["image_ids"].size(0),
            "tags": cache_data["tags"].size(0),
            "ratings": cache_data["ratings"].size(0),
            "dirs": cache_data["dirs"].size(0),
        }
        # Quick sanity check: all tensors should have some data
        if any(s == 0 for s in tensor_sizes.values()):
            logger.error(f"Empty tensor detected in cache: {tensor_sizes}")
            del cache_data
            return None

        # Decode string tensors - process each field separately to reduce peak memory
        logger.info("  Decoding tensors to strings...")
        decode_start = time.time()

        # Decode each field and immediately free intermediate bytes to reduce memory
        # Use errors='surrogateescape' to match encoding and filter empty strings from trailing newlines
        image_ids_bytes = cache_data["image_ids"].numpy().tobytes()
        image_ids = [s for s in image_ids_bytes.decode("utf-8", errors="surrogateescape").split("\n") if s]
        del image_ids_bytes

        tags_bytes = cache_data["tags"].numpy().tobytes()
        tags_json = [s for s in tags_bytes.decode("utf-8", errors="surrogateescape").split("\n") if s]
        del tags_bytes

        ratings_bytes = cache_data["ratings"].numpy().tobytes()
        ratings = [s for s in ratings_bytes.decode("utf-8", errors="surrogateescape").split("\n") if s]
        del ratings_bytes

        dirs_bytes = cache_data["dirs"].numpy().tobytes()
        dirs = [s for s in dirs_bytes.decode("utf-8", errors="surrogateescape").split("\n") if s]
        del dirs_bytes

        # Force garbage collection after heavy allocations
        gc.collect()

        # Validate array lengths match (critical for data integrity)
        array_lengths = {
            "image_ids": len(image_ids),
            "tags": len(tags_json),
            "ratings": len(ratings),
            "dirs": len(dirs),
        }
        if len(set(array_lengths.values())) != 1:
            logger.error(f"Array length mismatch in cache: {array_lengths}")
            return None  # Force rebuild

        decode_elapsed = time.time() - decode_start
        total_items = array_lengths["image_ids"]
        logger.info(f"  Decoded {total_items:,} items in {decode_elapsed:.1f}s, deserializing tags...")

        # Choose JSON parser (orjson is 5-10x faster)
        if HAS_ORJSON:
            json_loads = orjson.loads
        else:
            json_loads = json.loads

        # Reconstruct items list with progress logging
        items = []
        failed_indices = []
        progress_interval = max(100000, total_items // 10)  # Log every 10% or 100K items
        deserialize_start = time.time()

        for i in range(total_items):
            try:
                items.append({
                    "image_id": image_ids[i],
                    "tags": json_loads(tags_json[i]),
                    "rating": ratings[i],
                    "dir": Path(dirs[i]),
                })
            except Exception as e:
                failed_indices.append(i)
                if len(failed_indices) <= 10:  # Log first 10 errors for debugging
                    logger.warning(f"Failed to deserialize item {i}: {e}")
                continue

            # Progress logging
            if (i + 1) % progress_interval == 0:
                pct = (i + 1) * 100 // total_items
                elapsed_deser = time.time() - deserialize_start
                rate = (i + 1) / elapsed_deser if elapsed_deser > 0 else 0
                remaining = (total_items - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Deserializing: {i + 1:,}/{total_items:,} ({pct}%) "
                    f"@ {rate:,.0f} items/s, ~{remaining:.0f}s remaining"
                )

        # Check error rate - zero tolerance for deserialization errors
        # Any failed item could cause training crashes later
        if failed_indices:
            error_rate = len(failed_indices) / total_items
            logger.error(
                f"Deserialization failed for {len(failed_indices):,} items ({error_rate:.2%}). "
                "Cache will be rebuilt to ensure data integrity."
            )
            # Zero tolerance: any deserialization error triggers rebuild
            return None

        elapsed = time.time() - start_time
        logger.info(
            f"Loaded {len(items):,} items from metadata cache in {elapsed:.1f}s"
            + (f" (using orjson)" if HAS_ORJSON else "")
        )

        return items

    except Exception as e:
        logger.error(f"Failed to load metadata cache: {e}", exc_info=True)
        return None


def try_load_metadata_cache(
    root_dir: Path,
    json_files: List[Path],
    force_rebuild: bool = False,
    num_workers: int = 16,
    staleness_check_samples: int = 100,
    logger: Optional[logging.Logger] = None
) -> Optional[List[Dict[str, Any]]]:
    """Try to load metadata from cache, building if necessary.

    Main entry point for metadata caching. Attempts to load from cache,
    validates freshness, and rebuilds if needed.

    Args:
        root_dir: Dataset root directory
        json_files: List of JSON file paths
        force_rebuild: Force cache rebuild even if valid
        num_workers: Number of parallel workers for cache building
        staleness_check_samples: Number of files to sample for staleness check
        logger: Logger instance (creates default if None)

    Returns:
        List of parsed metadata dicts, or None on failure (triggers fallback)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        cache_path = _metadata_cache_path(root_dir)
        expected_count = len(json_files)

        # Check if we should use cache
        cache_exists = cache_path.exists()
        should_rebuild = force_rebuild

        if cache_exists and not force_rebuild:
            # Check if cache is stale
            if _is_cache_stale(cache_path, json_files, staleness_check_samples, logger):
                logger.info("Metadata cache is stale, rebuilding...")
                should_rebuild = True
            else:
                # Try to load cache (v2.0: with file list validation)
                items = _load_cached_metadata(cache_path, expected_count, json_files, logger)
                if items is not None:
                    return items
                else:
                    logger.warning("Failed to load cache, rebuilding...")
                    should_rebuild = True

        # Build cache if needed
        if should_rebuild or not cache_exists:
            # Delete old cache if it exists
            if cache_exists:
                try:
                    cache_path.unlink()
                except Exception:
                    pass

            # Build new cache
            items = _build_metadata_cache(
                cache_path, json_files, num_workers, logger
            )
            return items

        # Should not reach here
        return None

    except Exception as e:
        logger.error(f"Metadata cache error: {e}", exc_info=True)
        # Return None to trigger fallback to sequential parsing
        return None
