"""
Metadata cache for SidecarJsonDataset.

Caches parsed JSON metadata (image_id, tags, rating, dir, json_stem, filename)
to avoid re-parsing millions of JSON files on every training run.

Uses PyArrow IPC format for zero-copy memory-mapped access across workers.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to use orjson for faster JSON parsing (5-10x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# Import required functions from utils
from utils.path_utils import sanitize_identifier
from utils.metadata_ingestion import parse_tags_field

# PyArrow for zero-copy metadata cache (memory-mapped, shared across workers)
try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    HAS_PYARROW = True
except ImportError:
    pa = None
    pa_ipc = None
    HAS_PYARROW = False

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
_ARROW_CACHE_VERSION = "2.0"  # Arrow IPC format (added json_stem column)
# Fraction of sidecar JSON files that may fail to parse before _build_arrow_cache
# refuses to persist the result. Per-file failures are logged and skipped, but the
# cache is built once and reused for the whole run, so a systematic parse problem
# would otherwise shrink the training corpus silently and permanently. Sized to
# absorb a handful of genuinely corrupt sidecars in a multi-million-file corpus
# without absorbing a naming-convention or filesystem-level failure.
_MAX_PARSE_DROP_RATIO = 0.001  # 0.1%


class ArrowCacheTruncatedError(RuntimeError):
    """Raised when metadata parsing lost enough rows that the result is unusable.

    Deliberately an exception rather than a falsy return. Returning False here
    makes ``try_load_arrow_cache`` return None, which ``SidecarJsonDataset``
    treats as "PyArrow unavailable" and answers by falling back to the sequential
    JSON parser -- which applies the SAME parse logic with NO drop gate at all.
    A truncation that this class exists to catch would be silently re-created
    there, after an expensive full re-parse. It must propagate.
    """
# NOTE: newly built caches also include a "filename" column (exact image file
# name from the sidecar JSON, used to skip per-sample extension probing).
# Intentionally NOT a version bump: readers treat the column as optional, so
# existing caches stay valid — they just don't get the path-resolution speedup.


def _compute_file_list_hash(json_files: List[Path]) -> str:
    """Compute stable hash of json file list for cache validation.

    For large datasets (>3000 files), samples first 1000, middle 1000, and last 1000
    files for performance while maintaining good coverage.

    IMPORTANT: Sampling is done BEFORE path resolution to avoid O(n) filesystem
    calls on huge datasets (5M+ files would take 10+ minutes otherwise).
    """
    total = len(json_files)

    # Sample FIRST (before any resolve() calls) for large datasets
    if total > 3000:
        # Sort indices by string path for deterministic sampling
        sorted_indices = sorted(range(total), key=lambda i: str(json_files[i]))
        mid = total // 2
        sample_indices = sorted_indices[:1000] + sorted_indices[mid-500:mid+500] + sorted_indices[-1000:]
        # Only resolve the sampled paths (3000 max instead of 5M+)
        sample_paths = [str(json_files[i].resolve()) for i in sample_indices]
    else:
        # Small dataset: resolve all
        sample_paths = sorted(str(p.resolve()) for p in json_files)

    combined = "\n".join(sample_paths)
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
    """Sample files from different directories for better coverage.

    For huge datasets (>1M files), uses simple random sampling to avoid
    O(n) iteration overhead. For smaller datasets, uses stratified sampling
    across directories for better coverage.
    """
    if sample_size >= len(json_files):
        return json_files

    total = len(json_files)

    # Deterministic RNG so the staleness verdict is reproducible run-to-run. An
    # unseeded random.sample() could draw different files each launch, letting a
    # stale cache pass on one run and fail on the next (the file-list hash samples
    # deterministically by sorted path for the same reason).
    rng = random.Random(42)

    # For huge datasets (>1M files), use simple random sampling
    # Stratified grouping requires O(n) iteration which is too slow for 5M+ files
    # Random sampling is O(sample_size) and provides sufficient coverage for staleness checks
    if total > 1_000_000:
        indices = rng.sample(range(total), sample_size)
        return [json_files[i] for i in indices]

    # For smaller datasets, use stratified sampling across directories
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
        samples.extend(rng.sample(dir_files, min(dir_sample_size, len(dir_files))))

    # Ensure we don't exceed sample_size due to rounding
    return samples[:sample_size]


def _arrow_cache_path(root: Path) -> Path:
    """Return cache file path for Arrow IPC metadata cache.

    Files live under ./logs/metadata_cache/<sha1(root)>.metadata.arrow
    """
    cache_dir = _PROJ_ROOT / "logs" / "metadata_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{key}.metadata.arrow"


def _arrow_meta_path(root: Path) -> Path:
    """Return path for Arrow cache metadata file (.arrow.meta)."""
    return _arrow_cache_path(root).with_suffix(".arrow.meta")


def _parse_json_batch(
    json_files: List[Path],
    logger: logging.Logger
) -> Tuple[List[Dict[str, Any]], int]:
    """Parse a batch of JSON files (worker function).

    Args:
        json_files: List of JSON file paths to parse
        logger: Logger instance

    Returns:
        (parsed metadata dicts, count of files skipped for not being dicts).

        The non-dict count is reported separately because those files are not
        failures -- a list-shaped JSON under the data root is a manifest, not a
        malformed sidecar. Folding them into the parse-failure total would let
        the caller's drop gate hard-fail a corpus that is perfectly healthy, and
        blame it on filename conventions. The sequential fallback in
        dataset_loader.py excludes them for the same reason.
    """
    # Use orjson if available (5-10x faster than stdlib json)
    if HAS_ORJSON:
        def load_json(path: Path):
            return orjson.loads(path.read_bytes())
    else:
        def load_json(path: Path):
            return json.loads(path.read_text(encoding="utf-8"))

    items = []
    non_dict_skips = 0
    for jp in json_files:
        try:
            data = load_json(jp)
            # Skip if data is not a dict (e.g., manifest files are lists)
            if not isinstance(data, dict):
                non_dict_skips += 1
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
                "json_stem": jp.stem,  # Original JSON filename stem for path reconstruction
                "filename": Path(fname).name,  # Exact image filename (skips extension probing)
            })
        except Exception as e:
            # Log warning but continue processing
            logger.warning(f"Failed to parse {jp}: {e}")
            continue

    return items, non_dict_skips


# =============================================================================
# Arrow IPC Cache Functions (Zero-Copy, Memory-Mapped)
# =============================================================================


def _load_arrow_cache(
    cache_path: Path,
    logger: Optional[logging.Logger] = None
) -> Optional["pa.Table"]:
    """Load Arrow IPC cache as memory-mapped table.

    The returned table is backed by a memory-mapped file, meaning:
    - No data is copied to RAM on load (just sets up virtual memory mapping)
    - Multiple processes can share the same physical memory pages
    - Accessing any row triggers a page fault that loads just that data

    Args:
        cache_path: Path to the Arrow IPC file
        logger: Optional logger

    Returns:
        PyArrow Table (memory-mapped), or None on failure
    """
    if not HAS_PYARROW:
        if logger:
            logger.error("PyArrow not installed - cannot load Arrow cache")
        return None

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        start_time = time.time()

        # Memory-map the file - this is nearly instant (no data copied)
        source = pa.memory_map(str(cache_path), 'r')
        reader = pa_ipc.open_file(source)
        table = reader.read_all()

        elapsed = time.time() - start_time
        logger.info(
            f"Loaded Arrow cache: {len(table):,} items in {elapsed:.2f}s "
            f"(memory-mapped, {cache_path.stat().st_size / 1e6:.1f} MB)"
        )
        return table

    except Exception as e:
        logger.error(f"Failed to load Arrow cache: {e}")
        return None


def _build_arrow_cache(
    json_files: List[Path],
    cache_path: Path,
    num_workers: int = 16,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Build Arrow IPC metadata cache from JSON files.

    Uses parallel JSON parsing (reuses existing _parse_json_batch) and
    writes to Arrow IPC format for memory-mapped access.

    Args:
        json_files: List of JSON file paths to parse
        cache_path: Path where Arrow cache will be saved
        num_workers: Number of parallel workers for JSON parsing
        logger: Optional logger

    Returns:
        True on success, False on failure
    """
    if not HAS_PYARROW:
        if logger:
            logger.error("PyArrow not installed - cannot build Arrow cache")
        return False

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        total_files = len(json_files)
        logger.info(f"Building Arrow metadata cache from {total_files:,} JSON files...")
        start_time = time.time()

        # --- Phase 1: Parse JSON files in parallel ---
        # Reuse existing parallel parsing logic
        # Determine chunk size based on dataset size
        if total_files < 100000:
            chunk_size = 1024
        elif total_files < 1000000:
            chunk_size = 4096
        else:
            chunk_size = 8192

        chunks = [json_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
        all_items = []

        logger.info(f"  Parsing {total_files:,} files in {len(chunks)} chunks using {num_workers} workers...")
        parse_start = time.time()

        failed_chunks = 0
        non_dict_skips = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_parse_json_batch, chunk, logger) for chunk in chunks]

            for i, future in enumerate(as_completed(futures)):
                try:
                    items, chunk_non_dict = future.result()
                    all_items.extend(items)
                    non_dict_skips += chunk_non_dict
                except Exception as e:
                    # A whole chunk is up to `chunk_size` files. Losing one is a
                    # different failure class from a per-file parse error and is
                    # never acceptable: the cache is built once per run and reused,
                    # so a transient failure here would silently shrink the training
                    # corpus for this run and every future one. Counted here and
                    # turned into a hard failure below.
                    failed_chunks += 1
                    logger.error(f"Chunk parsing failed (chunk of up to {chunk_size:,} files): {e}")

                # Progress logging
                if (i + 1) % max(1, len(chunks) // 10) == 0:
                    pct = (i + 1) * 100 // len(chunks)
                    logger.info(f"  Parsing progress: {i + 1}/{len(chunks)} chunks ({pct}%)")

        parse_elapsed = time.time() - parse_start
        logger.info(f"  Parsed {len(all_items):,} items in {parse_elapsed:.1f}s")

        # Surface systematic data loss: sanitize_identifier / parse failures are
        # caught per-file with a warning + continue, so a naming-convention violation
        # (spaces, unicode, parentheses) can silently drop every such image from the
        # cache (and thus from training) with only buried per-file warnings. One
        # aggregate warning makes the loss visible at build time.
        #
        # Non-dict files (manifests and other list-shaped JSON that the sidecar
        # glob swept up) are NOT losses — they were never sidecars. They are
        # removed from both sides of the ratio so the gate below cannot hard-fail
        # a healthy corpus and blame it on filename conventions. The sequential
        # fallback in dataset_loader.py treats them the same way.
        considered = total_files - non_dict_skips
        dropped = considered - len(all_items)
        drop_ratio = (dropped / considered) if considered else 0.0
        if non_dict_skips > 0:
            logger.info(
                f"  Ignored {non_dict_skips:,} non-sidecar JSON file(s) (list-shaped, e.g. "
                f"manifests). These are not counted as parse failures."
            )
        if dropped > 0:
            logger.warning(
                f"  {dropped:,} of {considered:,} sidecar JSON files ({drop_ratio * 100:.2f}%) were "
                f"dropped during parse and are absent from the cache (and from training). "
                f"Common cause: filenames violating the id pattern (spaces/unicode/parentheses). "
                f"Inspect the per-file 'Failed to parse' warnings above if this ratio is unexpected."
            )

        # Refuse to persist a materially truncated cache. Warning alone is not
        # enough: this cache is written once and reused for the whole run (and
        # subsequent runs), so anything lost here is lost silently from every
        # epoch that follows. A whole-chunk failure is always fatal; per-file
        # drops are tolerated only up to _MAX_PARSE_DROP_RATIO, which is sized
        # to permit a handful of genuinely malformed sidecars but not a
        # systematic naming-convention or filesystem problem.
        #
        # Checked BEFORE the `not all_items` guard below: if every chunk failed,
        # all_items is empty too, and the generic "no items parsed" message would
        # hide the actual cause.
        if failed_chunks > 0:
            raise ArrowCacheTruncatedError(
                f"{failed_chunks} of {len(chunks)} metadata parse chunk(s) failed outright, "
                f"so up to {failed_chunks * chunk_size:,} of {total_files:,} files are missing. "
                f"Accepting this would silently shrink the training corpus for this run and "
                f"every run that reuses the cache. Fix the underlying error and rebuild."
            )

        # The absolute floor matters as much as the ratio. On a small dataset the
        # ratio is dominated by the single-file quantum -- 1 bad sidecar out of 64
        # is 1.56%, which a bare ratio test would treat as a systematic failure and
        # hard-fail permanently. This project keeps several such small corpora (the
        # Anima canary set and assorted probe splits are 6-64 files). The floor is
        # the same one the staleness gate and the sequential-parse gate use, so all
        # three agree on one shape; at production scale it is exactly equivalent to
        # the bare ratio (both trip at dropped >= 5922 for 5.9M files).
        if dropped > max(100, int(considered * _MAX_PARSE_DROP_RATIO)):
            raise ArrowCacheTruncatedError(
                f"{dropped:,} of {considered:,} sidecar JSON files ({drop_ratio * 100:.2f}%) "
                f"failed to parse, exceeding the {_MAX_PARSE_DROP_RATIO * 100:.2f}% tolerance "
                f"(minimum 100 files). "
                f"This is a systematic data problem, not incidental corruption — inspect the "
                f"per-file 'Failed to parse' warnings above. Raise "
                f"utils.metadata_cache._MAX_PARSE_DROP_RATIO only if the loss is understood "
                f"and intentional."
            )

        if not all_items:
            logger.error("No items parsed from JSON files")
            return False

        # --- Phase 2: Build PyArrow Table ---
        logger.info("  Building Arrow table...")
        arrow_start = time.time()

        # Extract columns
        image_ids = [item["image_id"] for item in all_items]
        tags_lists = [item["tags"] for item in all_items]
        ratings = [item["rating"] for item in all_items]
        dirs = [item["dir"] for item in all_items]
        json_stems = [item["json_stem"] for item in all_items]
        filenames = [item["filename"] for item in all_items]

        # Create PyArrow arrays
        # Tags use list type for variable-length arrays
        table = pa.table({
            "image_id": pa.array(image_ids, type=pa.string()),
            "tags": pa.array(tags_lists, type=pa.list_(pa.string())),
            "rating": pa.array(ratings, type=pa.string()),
            "dir": pa.array(dirs, type=pa.string()),
            "json_stem": pa.array(json_stems, type=pa.string()),
            "filename": pa.array(filenames, type=pa.string()),
        })

        arrow_elapsed = time.time() - arrow_start
        logger.info(f"  Built Arrow table in {arrow_elapsed:.1f}s")

        # --- Phase 3: Write to IPC file ---
        logger.info("  Writing Arrow IPC file...")
        write_start = time.time()

        # Write to temp file first, then atomic rename
        temp_path = cache_path.with_suffix(".arrow.tmp")
        lock_path = cache_path.with_suffix(".arrow.lock")

        def _do_arrow_write():
            """Perform the atomic write operations."""
            try:
                with pa.OSFile(str(temp_path), 'wb') as sink:
                    with pa_ipc.new_file(sink, table.schema) as writer:
                        # Write in batches to control memory
                        batch_size = 100000
                        for i in range(0, len(table), batch_size):
                            batch = table.slice(i, min(batch_size, len(table) - i))
                            writer.write_table(batch)

                # Drop the previous meta before publishing the new table. The
                # meta is written in Phase 4, so a kill between the rename below
                # and that write would otherwise leave the OLD meta describing
                # the NEW table. Removing it first makes the crash state
                # "table with no meta", which the caller already treats as
                # invalid and rebuilds. (Nothing is deleted up front any more —
                # this runs only once the replacement table is fully written.)
                cache_path.with_suffix(".arrow.meta").unlink(missing_ok=True)

                # Atomic rename
                os.replace(temp_path, cache_path)

            except Exception:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise

        # Use file locking if available (prevents concurrent write corruption)
        if HAS_FILELOCK:
            lock = filelock.FileLock(lock_path)
            try:
                # Fast path: try non-blocking first (timeout=0)
                # If another process is building, we'll wait on the slow path
                lock.acquire(timeout=0)
                logger.debug("Acquired cache lock immediately (non-blocking)")
            except filelock.Timeout:
                # Slow path: another process is building, wait with reduced timeout
                # 30s is sufficient for most cache writes; 600s was excessive
                logger.info("Cache lock held by another process, waiting up to 30s...")
                try:
                    lock.acquire(timeout=30)
                except filelock.Timeout:
                    logger.error(
                        f"Timeout acquiring cache lock after 30s - another process may be stuck. "
                        f"Lock file: {lock_path}"
                    )
                    return False

            try:
                _do_arrow_write()
            finally:
                lock.release()
        else:
            _do_arrow_write()

        write_elapsed = time.time() - write_start
        file_size_mb = cache_path.stat().st_size / 1e6
        logger.info(f"  Wrote Arrow IPC file in {write_elapsed:.1f}s ({file_size_mb:.1f} MB)")

        # --- Phase 4: Write metadata file ---
        meta_path = cache_path.with_suffix(".arrow.meta")
        meta_data = {
            "version": _ARROW_CACHE_VERSION,
            "count": len(table),
            # NOTE: recorded for provenance/future use but intentionally NOT consulted
            # by _is_arrow_cache_stale() - the cache is built once per run with no
            # mid-run sidecar edits (build-once policy), so a file-list-hash staleness
            # gate would never fire. Kept so the meta stays self-describing; do not
            # treat its presence as an active integrity check.
            "file_list_hash": _compute_file_list_hash(json_files),
            "file_count": len(json_files),
            # Files that failed to parse and are therefore absent from the table.
            # `file_count` is the INPUT count, so on its own it cannot reveal a
            # truncated cache to a later run — _is_arrow_cache_stale() compares
            # it against the current file list and would see a perfect match even
            # if half the rows were dropped at build time. Recording the drop
            # explicitly gives the staleness gate something to check.
            "parsed_count": len(all_items),
            "dropped_count": dropped,
            # Sidecars actually eligible to become rows: file_count minus the
            # list-shaped/manifest JSONs the glob swept up. The staleness gate
            # must measure truncation against this, not file_count, or a corpus
            # with manifests looks permanently truncated and rebuilds forever.
            "considered_count": total_files - non_dict_skips,
            "created_at": time.time(),
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)

        total_elapsed = time.time() - start_time
        logger.info(
            f"Arrow metadata cache built: {len(table):,} items in {total_elapsed:.1f}s "
            f"({file_size_mb:.1f} MB)"
        )
        return True

    except ArrowCacheTruncatedError:
        # Raised by the drop gates above, from inside this same try block.
        # Letting the generic handler below convert it to `return False` would
        # defeat the gate entirely: the caller reads False as "cache unavailable"
        # and falls back to the ungated sequential parser.
        raise
    except Exception as e:
        logger.error(f"Failed to build Arrow cache: {e}", exc_info=True)
        return False


def _is_arrow_cache_stale(
    cache_path: Path,
    json_files: List[Path],
    staleness_check_samples: int = 100,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Check if Arrow cache is stale.

    Args:
        cache_path: Path to Arrow cache file
        json_files: Current list of JSON files
        staleness_check_samples: Base sample size for staleness check
        logger: Optional logger

    Returns:
        True if cache is stale, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        meta_path = cache_path.with_suffix(".arrow.meta")

        # Check meta file exists
        if not meta_path.exists():
            logger.info("Arrow cache meta file not found")
            return True

        # Load and validate meta
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # Version check
        if meta.get("version") != _ARROW_CACHE_VERSION:
            logger.info(f"Arrow cache version mismatch: {meta.get('version')} != {_ARROW_CACHE_VERSION}")
            return True

        # File count check (with tolerance)
        cached_count = meta.get("file_count", 0)
        current_count = len(json_files)
        tolerance = max(100, int(current_count * 0.001))  # 0.1% tolerance
        if abs(cached_count - current_count) > tolerance:
            logger.info(
                f"Arrow cache file count drift: {cached_count:,} cached vs {current_count:,} current "
                f"(tolerance: {tolerance})"
            )
            return True

        # Truncation check. `file_count` above is the INPUT file count, so a cache
        # whose rows were dropped at parse time matches it perfectly and would pass
        # the drift check forever.
        #
        # `count` has always held len(all_items) — the number of rows that actually
        # made it into the table — so this works on caches written before the
        # dedicated `parsed_count` key existed. That legacy population is precisely
        # the one that could be truncated (it predates the build-time gate), so it
        # must NOT be exempted. `parsed_count` is preferred when present purely
        # because it is unambiguous.
        parsed_count = meta.get("parsed_count", meta.get("count"))
        # Measure against eligible sidecars, not raw file count — see the
        # considered_count note where the meta is written. Legacy metas lack the
        # key and fall back to file_count, which is correct for them (the caches
        # on disk predate manifests being swept up and show zero non-dict files).
        considered_count = int(meta.get("considered_count", cached_count) or 0)
        if parsed_count is not None and considered_count > 0:
            missing = considered_count - int(parsed_count)
            if missing > max(100, int(considered_count * _MAX_PARSE_DROP_RATIO)):
                logger.warning(
                    f"Arrow cache is truncated: {missing:,} of {considered_count:,} sidecar "
                    f"file(s) ({missing / considered_count * 100:.2f}%) were dropped when it "
                    f"was built. Rebuilding rather than training on the reduced corpus."
                )
                return True

        # Sample-based mtime check (reuse existing logic)
        cache_mtime = cache_path.stat().st_mtime
        sample_size = _compute_sample_size(current_count, staleness_check_samples)
        samples = _stratified_sample(json_files, sample_size)

        # Check if any sampled file is newer than cache
        for jp in samples:
            try:
                if jp.stat().st_mtime > cache_mtime:
                    logger.info(f"Arrow cache stale: {jp} modified after cache")
                    return True
            except OSError:
                continue

        return False

    except Exception as e:
        logger.warning(f"Arrow staleness check failed: {e}")
        return True  # Assume stale on error


def try_load_arrow_cache(
    root_dir: Path,
    json_files: List[Path],
    force_rebuild: bool = False,
    num_workers: int = 16,
    staleness_check_samples: int = 100,
    logger: Optional[logging.Logger] = None
) -> Optional["pa.Table"]:
    """Load or build Arrow metadata cache.

    Main entry point for Arrow-based metadata caching. Attempts to load
    from existing cache, validates freshness, and rebuilds if needed.

    Args:
        root_dir: Dataset root directory
        json_files: List of JSON metadata files
        force_rebuild: Force cache rebuild even if valid
        num_workers: Workers for parallel JSON parsing
        staleness_check_samples: Base sample size for staleness check
        logger: Logger instance

    Returns:
        PyArrow Table (memory-mapped) on success, None on failure
    """
    if not HAS_PYARROW:
        if logger:
            logger.warning(
                "PyArrow not installed - falling back to legacy cache. "
                "Install with: pip install pyarrow>=14.0.0"
            )
        return None

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        cache_path = _arrow_cache_path(root_dir)
        cache_exists = cache_path.exists()

        # Try existing cache
        if cache_exists and not force_rebuild:
            if not _is_arrow_cache_stale(cache_path, json_files, staleness_check_samples, logger):
                table = _load_arrow_cache(cache_path, logger)
                if table is not None:
                    # Validate row count against metadata
                    meta_path = cache_path.with_suffix(".arrow.meta")
                    if not meta_path.exists():
                        logger.warning(
                            "Arrow cache metadata file missing - treating as invalid"
                        )
                        # Fall through to rebuild
                    else:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        expected_count = meta.get("count", 0)
                        if len(table) != expected_count:
                            logger.warning(
                                f"Arrow cache count mismatch: {len(table)} vs {expected_count}"
                            )
                            # Fall through to rebuild
                        else:
                            return table

            logger.info("Arrow cache is stale or invalid, rebuilding...")

        # Build new cache. The stale cache is deliberately NOT unlinked first:
        # _build_arrow_cache writes to a .tmp file and os.replace()s it into
        # position, so the write is already atomic, and deleting up front would
        # destroy a usable cache in exchange for nothing whenever the rebuild
        # fails.
        success = _build_arrow_cache(json_files, cache_path, num_workers, logger)
        if success:
            return _load_arrow_cache(cache_path, logger)
        else:
            return None

    except ArrowCacheTruncatedError:
        # Must NOT be swallowed into `return None`. The caller reads None as
        # "Arrow unavailable" and falls back to the sequential JSON parser,
        # which re-applies the same parse logic with no drop gate — silently
        # reproducing the exact truncation this error exists to prevent.
        raise
    except Exception as e:
        logger.error(f"Arrow cache error: {e}", exc_info=True)
        return None
