"""
Near-Duplicate Cluster Finder - Scalable version for millions of images.

Two-phase approach for finding clusters of similar images:
  Phase 1 (Scan): Hash all shards and save to persistent cache (resumable)
  Phase 2 (Cluster): Load cached hashes, build clusters, export to JSON for review

Uses perceptual hashing (dHash) with LSH-based bucketing for efficient similarity detection,
Union-Find for transitive clustering, and multi-criteria selection to keep the best image
from each cluster.

Selection priority:
  1. Highest resolution (width x height)
  2. Tiebreaker: Highest tag count (from .json or .txt sidecar)
  3. Tiebreaker: Largest file size
  4. Final fallback: Deterministic random (MD5 hash of path)

Usage:
    # Phase 1: Scan and hash all shards (resumable)
    python find_near_dupes_cluster.py --phase 1 --data-root "L:\\Dab\\Dab" --workers 16

    # Resume Phase 1 after interruption
    python find_near_dupes_cluster.py --phase 1 --resume

    # Phase 2: Build clusters and export JSON for review
    python find_near_dupes_cluster.py --phase 2 --threshold 0.95

    # After reviewing dedup_clusters.json, generate deletion list
    python find_near_dupes_cluster.py --generate-deletions --output near_duplicate_deletions.txt

    # Full run (both phases, then wait for review)
    python find_near_dupes_cluster.py --full --data-root "L:\\Dab\\Dab" --threshold 0.95 --workers 16
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import random
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import multiprocessing

import numpy as np
from PIL import Image, ImageFile
import imagehash
from tqdm import tqdm

# GPU acceleration (optional)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Try to use orjson for faster JSON parsing
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# Import cache utilities
from .dedup_cache import (
    save_shard_hashes,
    load_shard_hashes,
    load_progress,
    save_progress,
    save_clusters,
    load_clusters,
    iter_cached_shard_metadata,
    get_cache_dir,
    get_clusters_path,
)

# GPU Hamming distance acceleration (optional)
try:
    from .gpu_hamming import GPUHammingProcessor, check_gpu_available, create_gpu_processor
    HAS_GPU_HAMMING = True
except ImportError:
    HAS_GPU_HAMMING = False
    GPUHammingProcessor = None

# Image extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}

# Pre-computed popcount table for fast Hamming distance (byte -> bit count)
POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


class LazyPathResolver:
    """
    Memory-efficient path storage using (shard_idx, local_idx) references.
    Saves ~10GB for 5.5M images by avoiding Python string storage overhead.
    """

    def __init__(self, total_images: int):
        # Store (shard_idx, local_idx) pairs as uint32 - 8 bytes vs ~200 bytes per path
        self.path_refs = np.zeros((total_images, 2), dtype=np.uint32)
        self.shard_base_paths = []  # Shard base paths (only ~800 entries)
        self.shard_rel_paths = []   # References to shard's relative path arrays
        self.current_idx = 0

    def add_shard(self, shard_base_path: str, relative_paths: list, count: int):
        """Add paths from a shard using references instead of copying strings."""
        shard_idx = len(self.shard_base_paths)
        self.shard_base_paths.append(shard_base_path)
        self.shard_rel_paths.append(relative_paths)

        end_idx = self.current_idx + count
        self.path_refs[self.current_idx:end_idx, 0] = shard_idx
        self.path_refs[self.current_idx:end_idx, 1] = np.arange(count, dtype=np.uint32)
        self.current_idx = end_idx

    def resolve(self, idx: int) -> str:
        """Lazily resolve a single path from (shard_idx, local_idx) reference."""
        shard_idx, local_idx = self.path_refs[idx]
        return os.path.join(self.shard_base_paths[shard_idx], self.shard_rel_paths[shard_idx][local_idx])

    def __len__(self):
        return self.current_idx

    def __getitem__(self, idx: int) -> str:
        """Support indexing like a list: path_resolver[idx]"""
        return self.resolve(idx)


def _hash_work_indices(indices: np.ndarray) -> int:
    """
    Hash sorted indices to 64-bit integer for deduplication.
    Uses Python's built-in hash on bytes - fast and memory efficient.
    Saves ~5-10GB by storing 8-byte hashes instead of frozenset/tuple objects.

    Collision probability is extremely low (<0.001%) for realistic bucket sizes.
    """
    sorted_idx = np.sort(indices)
    return hash(sorted_idx.tobytes())


def _get_1bit_neighbors(val: int, bit_width: int = 16) -> list:
    """Generate all 1-bit Hamming neighbors of a 16-bit bucket value."""
    return [val ^ (1 << bit_pos) for bit_pos in range(bit_width)]


def _generate_byte_permutations(num_bytes: int, num_perms: int, seed: int) -> list:
    """Generate reproducible byte index permutations for LSH."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(num_bytes) for _ in range(num_perms)]


def _compute_lsh_block_values_single_perm(hashes: np.ndarray, byte_perm: np.ndarray) -> np.ndarray:
    """
    Memory-efficient computation of 16-bit LSH bucket values for a SINGLE permutation.

    This avoids the memory explosion from fancy indexing with all K permutations at once.

    Args:
        hashes: (N, HASH_BYTE_COUNT) uint8 array
        byte_perm: Single permutation array of shape (HASH_BYTE_COUNT,)

    Returns:
        (N, NUM_BUCKETS) uint16 array of bucket values for this permutation
    """
    n = hashes.shape[0]
    num_buckets = hashes.shape[1] // 2

    # Apply single permutation: (N, HASH_BYTE_COUNT) - minimal memory overhead
    permuted = hashes[:, byte_perm]

    # Reshape to pair bytes: (N, NUM_BUCKETS, 2)
    reshaped = permuted.reshape(n, num_buckets, 2)

    # Combine byte pairs into 16-bit values in-place where possible: (N, NUM_BUCKETS)
    block_values = reshaped[:, :, 0].astype(np.uint16) << 8
    block_values |= reshaped[:, :, 1]

    del permuted, reshaped
    return block_values


def _count_bucket_sizes_worker(args):
    """
    Worker function for Pass 1c-a: Count bucket sizes for one permutation.
    Thread-safe - operates on independent data structures.
    """
    perm_idx, chunk_hashes, byte_perm, num_buckets = args
    block_values = _compute_lsh_block_values_single_perm(chunk_hashes, byte_perm)

    # Build count dictionaries for this permutation
    results = [{} for _ in range(num_buckets)]
    for block_idx in range(num_buckets):
        vals = block_values[:, block_idx]
        unique, counts = np.unique(vals, return_counts=True)
        bucket_dict = results[block_idx]
        for v, c in zip(unique, counts):
            bucket_dict[int(v)] = int(c)

    del block_values
    return perm_idx, results


def _fill_bucket_arrays_worker(args):
    """
    Worker function for Pass 1c-c: Fill bucket arrays for one permutation.
    Thread-safe - each permutation has independent bucket_assignments and offsets.

    Optimized: Uses sort-based grouping to minimize GIL contention.
    The key insight is that np.argsort runs entirely in C (releases GIL),
    while the previous approach had ~60K dict lookups per bucket holding the GIL.
    """
    (perm_idx, chunk_hashes, byte_perm, chunk_start, chunk_size,
     assignments_dict_list, offsets_dict_list, num_buckets) = args

    block_values = _compute_lsh_block_values_single_perm(chunk_hashes, byte_perm)
    chunk_indices = np.arange(chunk_size, dtype=np.int32) + chunk_start

    for block_idx in range(num_buckets):
        vals = block_values[:, block_idx]
        assignments_dict = assignments_dict_list[block_idx]
        offsets_dict = offsets_dict_list[block_idx]

        # Skip if no pre-allocated buckets for this block
        if not assignments_dict:
            continue

        # Sort chunk indices by bucket value - O(N log N) but runs entirely in C
        # This releases the GIL, allowing true parallelism across threads
        sort_order = np.argsort(vals)
        sorted_vals = vals[sort_order]
        sorted_chunk_indices = chunk_indices[sort_order]

        # Find boundaries between different bucket values using vectorized diff
        # This is O(N) and also runs in C
        boundaries = np.where(np.diff(sorted_vals) != 0)[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(sorted_vals)]])

        # Process each contiguous group - now only ~1K iterations (groups with count>=2)
        # instead of ~60K iterations (all unique values)
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            bucket_val = int(sorted_vals[start])

            arr = assignments_dict.get(bucket_val)
            if arr is not None:
                offset = offsets_dict[bucket_val]
                n_items = end - start
                arr[offset:offset + n_items] = sorted_chunk_indices[start:end]
                offsets_dict[bucket_val] = offset + n_items

    del block_values, chunk_indices
    return perm_idx


class UnionFind:
    """Disjoint Set Union (Union-Find) for efficient transitive clustering."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in same set
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def get_tag_count(image_path):
    """Get tag count from JSON or TXT sidecar file."""
    base_path = Path(image_path)

    # Try JSON sidecar first
    json_path = base_path.with_suffix('.json')
    if json_path.exists():
        try:
            if HAS_ORJSON:
                data = orjson.loads(json_path.read_bytes())
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            tags = data.get('tags', '')
            # Handle both list and comma-separated string formats
            if isinstance(tags, list):
                return len(tags)
            elif isinstance(tags, str) and tags:
                return len([t.strip() for t in tags.split(',') if t.strip()])
            elif tags:
                # Non-empty but unexpected type
                logging.debug(f"Unexpected tags type in {json_path}: {type(tags).__name__}")
            return 0
        except Exception as e:
            logging.debug(f"Failed to read JSON sidecar {json_path}: {e}")

    # Fall back to TXT sidecar (comma or newline separated)
    txt_path = base_path.with_suffix('.txt')
    if txt_path.exists():
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Handle both comma-separated and newline-separated tags
            if ',' in content:
                tags = [t.strip() for t in content.split(',') if t.strip()]
            else:
                tags = [t.strip() for t in content.split('\n') if t.strip()]
            return len(tags)
        except Exception as e:
            logging.debug(f"Failed to read TXT sidecar {txt_path}: {e}")

    return 0


def delete_image_and_sidecars(path):
    """Delete an image file and its associated sidecar files (.json, .txt)."""
    base_path = Path(path)
    deleted = []

    # Delete the image itself
    try:
        if base_path.exists():
            base_path.unlink()
            deleted.append(str(base_path))
    except Exception as e:
        logging.warning(f"Failed to delete image {path}: {e}")
        return deleted

    # Delete sidecar files
    for ext in ['.json', '.txt']:
        sidecar = base_path.with_suffix(ext)
        try:
            if sidecar.exists():
                sidecar.unlink()
                deleted.append(str(sidecar))
        except Exception as e:
            logging.debug(f"Failed to delete sidecar {sidecar}: {e}")

    return deleted


def hash_image_full(path, hash_size=16, hash_algorithms=('dhash',), delete_corrupt=True):
    """
    Hash a single image with full metadata using one or more perceptual hash algorithms.
    Returns (path, hash_bytes, resolution, tag_count, file_size) or None on failure.
    Thread-safe - no shared state.

    Args:
        path: Path to the image file
        hash_size: Size of the hash (default 16 = 256-bit per algorithm)
        hash_algorithms: Tuple of algorithms to use ('dhash', 'phash', 'ahash')
        delete_corrupt: If True, delete images with corrupt EXIF data (default True)

    Returns:
        Tuple of (path, hash_bytes, resolution, tag_count, file_size),
        "DELETED_CORRUPT_EXIF" if deleted, or None on failure.
        hash_bytes is concatenated bytes from all algorithms.
    """
    try:
        # Capture warnings during image loading to detect corrupt EXIF
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            with Image.open(path) as img:
                # Force load to trigger EXIF parsing
                img.load()
                resolution = img.width * img.height

                # Compute all requested hash algorithms
                hash_bytes_list = []
                hash_byte_count = (hash_size * hash_size) // 8

                for algo in hash_algorithms:
                    if algo == 'dhash':
                        h = imagehash.dhash(img, hash_size=hash_size)
                    elif algo == 'phash':
                        h = imagehash.phash(img, hash_size=hash_size)
                    elif algo == 'ahash':
                        h = imagehash.average_hash(img, hash_size=hash_size)
                    else:
                        raise ValueError(f"Unknown hash algorithm: {algo}")

                    h_int = int(str(h), 16)
                    hash_bytes_list.append(h_int.to_bytes(hash_byte_count, 'big'))

        # Check for corrupt EXIF warning
        for w in caught_warnings:
            if "Corrupt EXIF data" in str(w.message):
                if delete_corrupt:
                    logging.info(f"Deleting image with corrupt EXIF: {path}")
                    deleted = delete_image_and_sidecars(path)
                    if deleted:
                        logging.debug(f"Deleted: {deleted}")
                    return "DELETED_CORRUPT_EXIF"  # Special marker for tracking
                else:
                    logging.warning(f"Skipping image with corrupt EXIF (use --delete-corrupt to remove): {path}")
                    return None

        # Concatenate all hash bytes
        combined_hash = b''.join(hash_bytes_list)

        tag_count = get_tag_count(path)

        try:
            file_size = os.path.getsize(path)
        except Exception as e:
            logging.debug(f"Failed to get file size for {path}: {e}")
            file_size = 0

        return (path, combined_hash, resolution, tag_count, file_size)
    except Exception as e:
        logging.debug(f"Failed to hash image {path}: {e}")
        return None


def discover_shards(data_root: Path):
    """Discover all shard directories in the data root."""
    shards = []
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and item.name.startswith('shard_'):
            shards.append(item)
    return shards


def discover_images_in_shard(shard_path: Path):
    """Discover all image files in a shard directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(shard_path.rglob(f'*{ext}'))
        images.extend(shard_path.rglob(f'*{ext.upper()}'))
    return sorted(set(images))


def run_phase1(args):
    """
    Phase 1: Scan all shards and compute hashes.
    Saves results to persistent cache for each shard.
    """
    print("=" * 70)
    print("PHASE 1: SCAN AND HASH IMAGES")
    print("=" * 70)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Data root does not exist: {data_root}")
        return False

    # Load or initialize progress
    progress = load_progress()

    # Ensure nested structures exist to prevent KeyError
    if "phase1_status" not in progress:
        progress["phase1_status"] = {}
    if "phase2_status" not in progress:
        progress["phase2_status"] = {}

    # Handle resume
    if args.resume and progress.get("phase1_status", {}).get("completed_shards"):
        completed = set(progress["phase1_status"]["completed_shards"])
        print(f"Resuming from previous run: {len(completed)} shards already completed")
    else:
        completed = set()
        # progress already loaded above, just update it
        progress["phase"] = 1
        progress["phase1_status"]["started_at"] = datetime.now().isoformat()
        progress["config"] = {
            "hash_size": args.hash_size,
            "threshold": args.threshold,
            "data_root": str(data_root),
            "hash_algorithms": list(args.hash_algorithms),
        }

    # Discover shards
    print(f"\nDiscovering shards in: {data_root}")
    shards = discover_shards(data_root)
    total_shards = len(shards)
    print(f"Found {total_shards} shards")

    if total_shards == 0:
        print(f"\nWARNING: No shard_* directories found in {data_root}")
        print("Expected directory structure: data_root/shard_00000/, shard_00001/, etc.")
        return False

    progress["phase1_status"]["total_shards"] = total_shards
    save_progress(progress)

    # Process each shard
    shards_to_process = [s for s in shards if s.name not in completed]
    print(f"Shards to process: {len(shards_to_process)}")
    print(f"Workers: {args.workers}")
    print("=" * 70)
    print()

    total_images = 0
    total_failed = 0
    total_deleted_corrupt = 0

    for shard_idx, shard_path in enumerate(shards_to_process):
        shard_name = shard_path.name
        print(f"\n[{len(completed) + shard_idx + 1}/{total_shards}] Processing {shard_name}...")

        # Discover images in this shard
        images = discover_images_in_shard(shard_path)
        if not images:
            print(f"  No images found, skipping")
            completed.add(shard_name)
            progress["phase1_status"]["completed_shards"] = list(completed)
            save_progress(progress)
            continue

        print(f"  Found {len(images):,} images")

        # Hash all images in parallel
        results = []
        failed = 0
        deleted_corrupt = 0

        # Determine whether to delete corrupt images (--keep-corrupt overrides --delete-corrupt)
        should_delete_corrupt = args.delete_corrupt and not args.keep_corrupt

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(hash_image_full, str(img), args.hash_size, args.hash_algorithms, should_delete_corrupt): img
                for img in images
            }

            with tqdm(total=len(futures), desc=f"  Hashing", unit="img", leave=False) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result == "DELETED_CORRUPT_EXIF":
                        deleted_corrupt += 1
                    elif result is not None:
                        results.append(result)
                    else:
                        failed += 1
                    pbar.update(1)

        status_parts = [f"Hashed: {len(results):,}", f"Failed: {failed:,}"]
        if deleted_corrupt > 0:
            status_parts.append(f"Deleted (corrupt EXIF): {deleted_corrupt:,}")
        print(f"  {', '.join(status_parts)}")
        total_images += len(results)
        total_failed += failed
        total_deleted_corrupt += deleted_corrupt

        if results:
            # Prepare arrays for saving
            # hash_byte_count = bytes per algorithm * number of algorithms
            hash_byte_count = ((args.hash_size * args.hash_size) // 8) * len(args.hash_algorithms)
            paths = []
            hashes = np.zeros((len(results), hash_byte_count), dtype=np.uint8)
            resolutions = np.zeros(len(results), dtype=np.int64)
            tag_counts = np.zeros(len(results), dtype=np.int32)
            file_sizes = np.zeros(len(results), dtype=np.int64)

            for i, (path, h_bytes, res, tags, size) in enumerate(results):
                # Store relative path from shard
                rel_path = str(Path(path).relative_to(shard_path))
                paths.append(rel_path)
                hashes[i] = np.frombuffer(h_bytes, dtype=np.uint8)
                resolutions[i] = res
                tag_counts[i] = tags
                file_sizes[i] = size

            # Save to cache
            if save_shard_hashes(shard_name, paths, hashes, resolutions, tag_counts, file_sizes, args.hash_size, args.hash_algorithms):
                print(f"  Saved to cache")
            else:
                print(f"  WARNING: Failed to save cache!")
                progress["phase1_status"].setdefault("failed_shards", []).append(shard_name)

        # Update progress
        completed.add(shard_name)
        progress["phase1_status"]["completed_shards"] = list(completed)
        save_progress(progress)

    # Summary
    print()
    print("=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Shards processed:  {len(completed):,}")
    print(f"Total images:      {total_images:,}")
    print(f"Failed to hash:    {total_failed:,}")
    if total_deleted_corrupt > 0:
        print(f"Deleted (corrupt): {total_deleted_corrupt:,}")
    print(f"Cache directory:   {get_cache_dir()}")
    print()
    print("Next step: Run Phase 2 to build clusters")
    print(f"  python {sys.argv[0]} --phase 2 --threshold {args.threshold}")
    print("=" * 70)

    return True


def _compute_hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Compute Hamming distance between two hash byte arrays."""
    xor = hash1 ^ hash2
    return int(POPCOUNT_TABLE[xor].sum())


def verify_borderline_clusters(
    clusters: dict,
    all_hashes: np.ndarray,
    all_paths,  # LazyPathResolver or list - supports indexing
    threshold: float,
    hash_size: int,
    primary_algorithms: tuple,
):
    """
    Re-verify clusters where matches are near the threshold boundary.
    Uses a secondary hash algorithm (pHash) to confirm borderline matches.

    Args:
        clusters: Dict mapping cluster root -> list of member indices
        all_hashes: Full hash array (N x hash_bytes)
        all_paths: LazyPathResolver or list - supports all_paths[idx] indexing
        threshold: Similarity threshold
        hash_size: Hash size per algorithm
        primary_algorithms: Tuple of algorithms used in primary hashing

    Returns:
        (verified_clusters, stats_dict) where stats_dict contains verification statistics
    """
    # Determine secondary algorithm (use pHash if not already used, else ahash)
    if 'phash' not in primary_algorithms:
        secondary_algo = 'phash'
    elif 'ahash' not in primary_algorithms:
        secondary_algo = 'ahash'
    else:
        secondary_algo = 'dhash'  # Fall back to dhash if both are used

    BORDERLINE_RATIO = 0.8  # Pairs with distance > 80% of max are "borderline"
    num_algorithms = len(primary_algorithms)
    BITS_PER_ALGO = hash_size * hash_size
    TOTAL_BITS = BITS_PER_ALGO * num_algorithms
    MAX_DISTANCE = round((1 - threshold) * TOTAL_BITS)
    borderline_threshold = int(MAX_DISTANCE * BORDERLINE_RATIO)

    # Secondary hash threshold (for single algorithm)
    secondary_max_distance = round((1 - threshold) * BITS_PER_ALGO)

    verified_clusters = {}
    stats = {
        'total_clusters': len(clusters),
        'borderline_clusters': 0,
        'clusters_split': 0,
        'clusters_verified': 0,
        'images_recomputed': 0,
    }

    print(f"  Borderline threshold: {borderline_threshold} bits (80% of {MAX_DISTANCE})")
    print(f"  Secondary algorithm: {secondary_algo}")

    for root, members in tqdm(clusters.items(), desc="Verifying borderline", unit="cluster"):
        if len(members) < 2:
            verified_clusters[root] = members
            continue

        # Check if any pair is borderline
        is_borderline = False
        for i, idx_a in enumerate(members):
            for idx_b in members[i + 1:]:
                dist = _compute_hamming_distance(all_hashes[idx_a], all_hashes[idx_b])
                if dist > borderline_threshold:
                    is_borderline = True
                    break
            if is_borderline:
                break

        if not is_borderline:
            verified_clusters[root] = members
            stats['clusters_verified'] += 1
            continue

        stats['borderline_clusters'] += 1

        # Compute secondary hash for all members
        secondary_hashes = {}
        hash_byte_count = BITS_PER_ALGO // 8

        for idx in members:
            path = all_paths[idx]
            try:
                with Image.open(path) as img:
                    img.load()
                    if secondary_algo == 'phash':
                        h = imagehash.phash(img, hash_size=hash_size)
                    elif secondary_algo == 'ahash':
                        h = imagehash.average_hash(img, hash_size=hash_size)
                    else:
                        h = imagehash.dhash(img, hash_size=hash_size)
                    h_int = int(str(h), 16)
                    secondary_hashes[idx] = np.frombuffer(
                        h_int.to_bytes(hash_byte_count, 'big'), dtype=np.uint8
                    )
                    stats['images_recomputed'] += 1
            except Exception:
                secondary_hashes[idx] = None

        # Rebuild cluster using Union-Find, requiring secondary hash agreement
        member_list = list(members)
        n = len(member_list)
        uf = UnionFind(n)

        for i in range(n):
            for j in range(i + 1, n):
                idx_a, idx_b = member_list[i], member_list[j]
                h_a, h_b = secondary_hashes.get(idx_a), secondary_hashes.get(idx_b)

                # If either hash failed, keep them together (benefit of doubt)
                if h_a is None or h_b is None:
                    uf.union(i, j)
                    continue

                # Check secondary hash distance
                dist = _compute_hamming_distance(h_a, h_b)
                if dist <= secondary_max_distance:
                    uf.union(i, j)

        # Extract sub-clusters
        sub_clusters = defaultdict(list)
        for i, idx in enumerate(member_list):
            sub_root = uf.find(i)
            sub_clusters[sub_root].append(idx)

        # Add sub-clusters to verified output
        if len(sub_clusters) > 1:
            stats['clusters_split'] += 1

        for sub_members in sub_clusters.values():
            if len(sub_members) >= 2:
                # Use first member as new root
                verified_clusters[sub_members[0]] = sub_members
            # Single-member sub-clusters are dropped (no longer duplicates)

    return verified_clusters, stats


def _compare_bucket_indices_inline(
    indices: np.ndarray,
    all_hashes: np.ndarray,
    max_distance: int,
    max_bucket_size: int,
) -> tuple:
    """
    Inline comparison of all pairs within a bucket (no shared memory needed).
    Uses NumPy broadcasting for fast Hamming distance calculation.
    Returns (pairs_list, comparison_count, was_sampled).
    """
    n = len(indices)
    if n < 2:
        return [], 0, False

    sampled = False
    working_indices = indices

    # Distance-aware sampling for large buckets
    if n > max_bucket_size:
        sampled = True
        bucket_hashes = all_hashes[indices]
        first_bytes = bucket_hashes[:, :4].astype(np.int32)
        centroid = first_bytes.mean(axis=0)
        quick_distances = np.abs(first_bytes - centroid).sum(axis=1)
        keep_idx = np.argsort(quick_distances)[:max_bucket_size]
        working_indices = indices[keep_idx]
        n = len(working_indices)

    # Get hashes for this bucket
    bucket_hashes = all_hashes[working_indices]

    # Vectorized pairwise XOR using broadcasting
    xor_result = bucket_hashes[:, np.newaxis, :] ^ bucket_hashes[np.newaxis, :, :]

    # Popcount via lookup table
    distances = POPCOUNT_TABLE[xor_result].sum(axis=2)

    # Extract upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    mask = distances[i_idx, j_idx] <= max_distance

    # Build pairs using original global indices
    pairs = []
    for i, j in zip(i_idx[mask], j_idx[mask]):
        a, b = int(working_indices[i]), int(working_indices[j])
        pairs.append((min(a, b), max(a, b)))

    return pairs, len(i_idx), sampled


def _load_shard_worker(args_tuple):
    """Worker function to load a single shard (for parallel loading)."""
    shard_name, shard_path_str = args_tuple
    shard_data = load_shard_hashes(shard_name)
    if shard_data is None:
        return (shard_name, None, shard_path_str)
    return (shard_name, shard_data, shard_path_str)


def run_phase2(args):
    """
    Phase 2: Load cached hashes, build clusters, export to JSON.
    Memory-optimized version using streaming bucket processing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Validate threshold range
    if not 0.0 <= args.threshold <= 1.0:
        print(f"ERROR: Threshold must be between 0.0 and 1.0 (got {args.threshold})")
        return False

    # Set random seed for reproducibility (used in large bucket sampling)
    random.seed(args.seed)

    print("=" * 70)
    print("PHASE 2: BUILD CLUSTERS")
    print("=" * 70)

    # Configuration - calculate dynamically based on hash_size and number of algorithms
    NUM_ALGORITHMS = len(args.hash_algorithms)
    BITS_PER_ALGO = args.hash_size * args.hash_size  # 256 for hash_size=16
    HASH_BITS = BITS_PER_ALGO * NUM_ALGORITHMS  # Total bits across all algorithms
    HASH_BYTE_COUNT = HASH_BITS // 8
    MAX_DISTANCE = round((1 - args.threshold) * HASH_BITS)
    NUM_BUCKETS = HASH_BYTE_COUNT // 2  # Number of 16-bit blocks

    print(f"Similarity threshold: {args.threshold:.0%} (max distance: {MAX_DISTANCE} bits)")
    print(f"Hash algorithms:     {', '.join(args.hash_algorithms)} ({NUM_ALGORITHMS} algorithm(s))")
    print(f"Hash size:           {args.hash_size}x{args.hash_size} = {BITS_PER_ALGO} bits per algo, {HASH_BITS} total")
    print(f"LSH config:          {args.num_permutations} permutation(s) x {NUM_BUCKETS} buckets")
    if args.multi_probe:
        print(f"Multi-probe:         Enabled (16 neighbors per bucket)")
    print(f"Workers:             {args.workers}")
    print("=" * 70)
    print()

    # Load progress to get config
    progress = load_progress()

    # Ensure nested structures exist to prevent KeyError
    if "phase1_status" not in progress:
        progress["phase1_status"] = {}
    if "phase2_status" not in progress:
        progress["phase2_status"] = {}

    if progress.get("config", {}).get("data_root"):
        data_root = Path(progress["config"]["data_root"])
    elif args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = None

    # Validate data_root is available and exists
    if data_root is None or not data_root.exists():
        print(f"ERROR: data_root not found or does not exist: {data_root}")
        print("       Either run Phase 1 first or specify a valid --data-root")
        return False

    # Pre-scan: Count total images, collect shard names, and validate hash_size
    # Uses metadata-only iterator to avoid loading full tensor data twice
    print("Pre-scan: Counting images for memory pre-allocation...")
    shard_names = []
    total_images = 0

    for shard_name, count, hash_size, hash_algorithms in iter_cached_shard_metadata():
        # Validate hash_size for every shard (catches mixed cache files from different runs)
        if hash_size != args.hash_size:
            print(f"ERROR: Shard '{shard_name}' has hash_size={hash_size}, "
                  f"expected {args.hash_size}")
            print("       Re-run Phase 1 with matching --hash-size or use the cached value")
            return False

        # Validate hash_algorithms match
        if hash_algorithms != args.hash_algorithms:
            print(f"ERROR: Shard '{shard_name}' has hash_algorithms={hash_algorithms}, "
                  f"expected {args.hash_algorithms}")
            print("       Re-run Phase 1 with matching --hash-algorithms or use the cached value")
            return False

        shard_names.append(shard_name)
        total_images += count

    print(f"  Found {len(shard_names)} shards, {total_images:,} images total")
    print()

    if total_images == 0:
        print("ERROR: No cached hashes found. Run Phase 1 first.")
        return False

    # Pre-allocate numpy arrays - MUCH more memory efficient than Python lists
    print("Pass 1a: Loading hashes from cache (parallel I/O)...")

    all_hashes = np.zeros((total_images, HASH_BYTE_COUNT), dtype=np.uint8)
    all_resolutions = np.zeros(total_images, dtype=np.int64)
    all_tag_counts = np.zeros(total_images, dtype=np.int32)
    all_file_sizes = np.zeros(total_images, dtype=np.int64)
    # Use LazyPathResolver instead of list - saves ~10GB for 5.5M images
    all_paths = LazyPathResolver(total_images)

    # Parallel shard loading using ThreadPoolExecutor (I/O bound, threads work well)
    # Use more threads than workers since this is I/O bound
    io_workers = min(64, args.workers * 4)

    # Prepare worker arguments
    worker_args = [(name, str(data_root / name)) for name in shard_names]

    # Load shards in parallel, collecting results
    shard_results = []
    failed_shards = []

    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        futures = {executor.submit(_load_shard_worker, arg): arg[0] for arg in worker_args}

        with tqdm(total=len(futures), desc="Loading shards", unit="shard") as pbar:
            for future in as_completed(futures):
                shard_name, shard_data, shard_path_str = future.result()
                if shard_data is None:
                    failed_shards.append(shard_name)
                    logging.warning(f"Failed to load shard cache: {shard_name}")
                else:
                    shard_results.append((shard_name, shard_data, shard_path_str))
                pbar.update(1)

    # Sort results by shard name to ensure deterministic ordering
    shard_results.sort(key=lambda x: x[0])

    # Copy loaded data into pre-allocated arrays (sequential, fast)
    print("Pass 1b: Copying to arrays...")
    current_idx = 0

    for shard_name, shard_data, shard_path_str in tqdm(shard_results, desc="Copying data", unit="shard"):
        n = len(shard_data)
        if n == 0:
            continue

        end_idx = current_idx + n

        # Copy data into pre-allocated arrays
        all_hashes[current_idx:end_idx] = shard_data.hashes
        all_resolutions[current_idx:end_idx] = shard_data.resolutions
        all_tag_counts[current_idx:end_idx] = shard_data.tag_counts
        all_file_sizes[current_idx:end_idx] = shard_data.file_sizes

        # Add paths via lazy resolver (stores references, not full strings)
        all_paths.add_shard(shard_path_str, shard_data.paths, n)

        current_idx = end_idx

    # Free shard_results memory
    del shard_results

    # Trim arrays if some shards failed or were empty
    if current_idx < total_images:
        all_hashes = all_hashes[:current_idx]
        all_resolutions = all_resolutions[:current_idx]
        all_tag_counts = all_tag_counts[:current_idx]
        all_file_sizes = all_file_sizes[:current_idx]
        total_images = current_idx

    # LSH index building - two-pass approach for memory efficiency
    # Pass 1c-a: Count bucket sizes (minimal memory)
    # Pass 1c-b: Pre-allocate arrays based on counts
    # Pass 1c-c: Fill arrays directly (no accumulation/concatenation)
    print("Pass 1c: Building LSH index (two-pass memory-efficient)...")

    K = args.num_permutations
    byte_perms = _generate_byte_permutations(HASH_BYTE_COUNT, K, args.seed) if K > 1 else [np.arange(HASH_BYTE_COUNT)]

    CHUNK_SIZE = 100_000

    # ========== Pass 1c-a: Count bucket sizes (threaded) ==========
    print(f"  Pass 1c-a: Counting bucket sizes ({K} threads)...")

    # Count phase: just count entries per bucket value (very compact - dict of ints)
    bucket_counts = [[{} for _ in range(NUM_BUCKETS)] for _ in range(K)]

    num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE
    lsh_workers = min(K, 16)  # Thread pool for permutation parallelism

    # Reuse single executor across all chunks to avoid creation overhead
    with ThreadPoolExecutor(max_workers=lsh_workers) as executor:
        with tqdm(total=num_chunks * K, desc="Counting buckets", unit="perm-chunk") as pbar:
            for chunk_start in range(0, total_images, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, total_images)
                chunk_hashes = all_hashes[chunk_start:chunk_end]

                # Process K permutations in parallel using threads
                futures = []
                for perm_idx in range(K):
                    worker_args = (perm_idx, chunk_hashes, byte_perms[perm_idx], NUM_BUCKETS)
                    futures.append(executor.submit(_count_bucket_sizes_worker, worker_args))

                # Collect results and merge into bucket_counts
                for future in as_completed(futures):
                    perm_idx, results = future.result()
                    # Merge results into main bucket_counts structure
                    for block_idx in range(NUM_BUCKETS):
                        target_dict = bucket_counts[perm_idx][block_idx]
                        for v, c in results[block_idx].items():
                            target_dict[v] = target_dict.get(v, 0) + c
                    pbar.update(1)

                del chunk_hashes

    # ========== Pass 1c-b: Pre-allocate arrays ==========
    print("  Pass 1c-b: Pre-allocating bucket arrays...")

    bucket_assignments = [[{} for _ in range(NUM_BUCKETS)] for _ in range(K)]
    bucket_offsets = [[{} for _ in range(NUM_BUCKETS)] for _ in range(K)]  # Fill offsets
    total_allocated = 0

    for perm_idx in range(K):
        for block_idx in range(NUM_BUCKETS):
            for val, count in bucket_counts[perm_idx][block_idx].items():
                if count >= 2:  # Only allocate for potential duplicates
                    bucket_assignments[perm_idx][block_idx][val] = np.empty(count, dtype=np.int32)
                    bucket_offsets[perm_idx][block_idx][val] = 0
                    total_allocated += count

    # Free counts after allocation
    del bucket_counts
    gc.collect()
    print(f"  Pre-allocated {total_allocated:,} indices across all buckets")

    # ========== Pass 1c-c: Fill bucket arrays (threaded) ==========
    print(f"  Pass 1c-c: Filling bucket arrays ({lsh_workers} threads)...")

    # Reuse single executor across all chunks to avoid creation overhead
    with ThreadPoolExecutor(max_workers=lsh_workers) as executor:
        with tqdm(total=num_chunks * K, desc="Filling buckets", unit="perm-chunk") as pbar:
            for chunk_start in range(0, total_images, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, total_images)
                chunk_hashes = all_hashes[chunk_start:chunk_end]
                chunk_size = chunk_end - chunk_start

                # Process K permutations in parallel using threads
                # Each permutation has independent bucket_assignments[perm_idx] - no race conditions
                futures = []
                for perm_idx in range(K):
                    worker_args = (
                        perm_idx, chunk_hashes, byte_perms[perm_idx],
                        chunk_start, chunk_size,
                        bucket_assignments[perm_idx], bucket_offsets[perm_idx],
                        NUM_BUCKETS
                    )
                    futures.append(executor.submit(_fill_bucket_arrays_worker, worker_args))

                # Wait for all permutations to complete
                for future in as_completed(futures):
                    future.result()  # Raises exception if worker failed
                    pbar.update(1)

                del chunk_hashes

    # Free offset tracking (no longer needed)
    del bucket_offsets
    gc.collect()

    if failed_shards:
        print(f"\n  WARNING: {len(failed_shards)} shard(s) failed to load and were excluded from duplicate detection!")
        print(f"           Run with --verbose to see details. Re-run Phase 1 to regenerate cache.")

    # Defensive validation: ensure path list matches array lengths
    if len(all_paths) != total_images:
        print(f"ERROR: Path count mismatch: {len(all_paths)} paths vs {total_images} images")
        print("       This indicates a bug in the shard loading logic.")
        return False

    print(f"  Loaded {total_images:,} images into memory-efficient arrays")
    print()

    # Initialize GPU if available and requested
    use_gpu = args.gpu and not args.no_gpu and HAS_GPU_HAMMING and HAS_TORCH
    gpu_processor = None
    all_hashes_gpu = None

    if use_gpu:
        available, gpu_info = check_gpu_available()
        if available:
            print(f"GPU acceleration: {gpu_info}")
            gpu_processor = create_gpu_processor(max_distance=MAX_DISTANCE)
            if gpu_processor is not None:
                # Transfer all hashes to GPU once (memory efficient)
                all_hashes_gpu = torch.from_numpy(all_hashes).to(
                    device=gpu_processor.device, dtype=torch.uint8, non_blocking=True
                )
                print(f"  Transferred {total_images:,} hashes to GPU ({all_hashes_gpu.numel() / 1e6:.1f} MB)")
                # Free CPU hash array after GPU transfer - saves ~700MB
                # BUT keep it if --verify-borderline is set (needs CPU hashes for borderline checking)
                torch.cuda.synchronize()
                if not args.verify_borderline:
                    hash_memory_mb = all_hashes.nbytes / 1e6
                    del all_hashes
                    gc.collect()
                    all_hashes = None  # Mark as freed
                    print(f"  Freed CPU hash memory ({hash_memory_mb:.1f} MB)")
                else:
                    print(f"  Keeping CPU hashes for --verify-borderline ({all_hashes.nbytes / 1e6:.1f} MB)")
            else:
                print("  GPU processor initialization failed, falling back to CPU")
                use_gpu = False
        else:
            print(f"GPU not available ({gpu_info}), using CPU")
            use_gpu = False
    else:
        if args.no_gpu:
            print("GPU acceleration: Disabled by --no-gpu flag")
        elif not HAS_GPU_HAMMING:
            print("GPU acceleration: gpu_hamming module not available")
        elif not HAS_TORCH:
            print("GPU acceleration: PyTorch not installed")
    print()

    # Pass 2: Find duplicate pairs using streaming bucket comparison
    # Process buckets directly without building massive work_items list
    print("Pass 2: Finding duplicate pairs via streaming LSH comparison...")

    MAX_BUCKET_SIZE = args.max_bucket_size

    # Count total buckets with 2+ images for progress bar
    total_buckets_to_process = 0
    for perm_idx in range(K):
        for block_idx in range(NUM_BUCKETS):
            for val, indices in bucket_assignments[perm_idx][block_idx].items():
                if len(indices) >= 2:
                    total_buckets_to_process += 1

    print(f"  Found {total_buckets_to_process:,} bucket values to compare")

    duplicate_pairs = set()
    comparisons = 0
    sampled_buckets = 0
    # Use hash-based dedup: 8 bytes per entry vs ~200 bytes for frozenset/tuple
    # Saves ~5-10GB for large datasets with millions of buckets
    processed_work_hashes = set()

    with tqdm(total=total_buckets_to_process, desc="Comparing pairs", unit="bucket") as pbar:
        for perm_idx in range(K):
            for block_idx in range(NUM_BUCKETS):
                bucket_dict = bucket_assignments[perm_idx][block_idx]

                for val, indices in bucket_dict.items():
                    if len(indices) < 2:
                        continue

                    # Multi-probe: merge 1-bit Hamming neighbors
                    if args.multi_probe:
                        work_indices = set(indices.tolist())
                        for neighbor_val in _get_1bit_neighbors(val):
                            neighbor_indices = bucket_dict.get(neighbor_val)
                            if neighbor_indices is not None:
                                work_indices.update(neighbor_indices.tolist())
                        if len(work_indices) < 2:
                            pbar.update(1)
                            continue
                        indices = np.array(list(work_indices), dtype=np.int32)
                        # Deduplicate using hash (memory-efficient)
                        work_hash = _hash_work_indices(indices)
                        if work_hash in processed_work_hashes:
                            pbar.update(1)
                            continue
                        processed_work_hashes.add(work_hash)
                    else:
                        # Simple case: just use this bucket's indices
                        # Deduplicate using hash (memory-efficient)
                        work_hash = _hash_work_indices(indices)
                        if work_hash in processed_work_hashes:
                            pbar.update(1)
                            continue
                        processed_work_hashes.add(work_hash)

                    # Compare pairs - use GPU if available, otherwise CPU
                    if use_gpu and gpu_processor is not None:
                        result = gpu_processor.find_similar_pairs(
                            indices, all_hashes_gpu, MAX_BUCKET_SIZE
                        )
                        pairs = result.pairs
                        comp_count = result.comparison_count
                        was_sampled = result.was_sampled
                    else:
                        pairs, comp_count, was_sampled = _compare_bucket_indices_inline(
                            indices, all_hashes, MAX_DISTANCE, MAX_BUCKET_SIZE
                        )

                    duplicate_pairs.update(pairs)
                    comparisons += comp_count
                    if was_sampled:
                        sampled_buckets += 1

                    pbar.update(1)

                # Clear processed bucket dict entries to free memory as we go
                bucket_dict.clear()

    # Clear bucket assignments to free memory
    del bucket_assignments
    del processed_work_hashes

    # Release GPU memory
    if use_gpu and all_hashes_gpu is not None:
        del all_hashes_gpu
        torch.cuda.empty_cache()
        if gpu_processor is not None:
            stats = gpu_processor.get_stats()
            print(f"  GPU stats: {stats['total_comparisons']:,} comparisons, {stats['total_pairs_found']:,} pairs found")

    if sampled_buckets > 0:
        print(f"  Note: {sampled_buckets:,} large bucket(s) used distance-aware sampling (limited to {MAX_BUCKET_SIZE} images)")

    print(f"  Comparisons made:    {comparisons:,}")
    print(f"  Duplicate pairs:     {len(duplicate_pairs):,}")
    print()

    # Phase 3: Build clusters with Union-Find
    print("Pass 3: Building clusters with Union-Find...")

    uf = UnionFind(total_images)
    for idx_a, idx_b in duplicate_pairs:
        uf.union(idx_a, idx_b)

    # Group by cluster root
    clusters = defaultdict(list)
    for idx in range(total_images):
        root = uf.find(idx)
        clusters[root].append(idx)

    # Filter to clusters with 2+ members
    clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

    total_in_clusters = sum(len(v) for v in clusters.values())
    print(f"  Unique clusters:     {len(clusters):,}")
    print(f"  Images in clusters:  {total_in_clusters:,}")
    print()

    # Optional: Verify borderline clusters with secondary hash
    if args.verify_borderline and len(clusters) > 0:
        print("Pass 3b: Verifying borderline clusters with secondary hash...")
        clusters, verify_stats = verify_borderline_clusters(
            clusters=clusters,
            all_hashes=all_hashes,
            all_paths=all_paths,
            threshold=args.threshold,
            hash_size=args.hash_size,
            primary_algorithms=args.hash_algorithms,
        )
        print(f"  Borderline clusters: {verify_stats['borderline_clusters']:,}")
        print(f"  Clusters split:      {verify_stats['clusters_split']:,}")
        print(f"  Images recomputed:   {verify_stats['images_recomputed']:,}")

        # Recalculate totals after verification
        total_in_clusters = sum(len(v) for v in clusters.values())
        print(f"  Final clusters:      {len(clusters):,}")
        print(f"  Final in clusters:   {total_in_clusters:,}")
    print()

    # Phase 4: Select best image per cluster and export
    print("Pass 4: Selecting best image per cluster and exporting JSON...")

    def get_score(idx):
        """Score tuple for comparison (higher is better)."""
        res = int(all_resolutions[idx])
        tags = int(all_tag_counts[idx])
        size = int(all_file_sizes[idx])
        path = all_paths[idx]
        random_val = int(hashlib.md5(path.encode()).hexdigest(), 16) % 1000000
        return (res, tags, size, random_val)

    clusters_output = []
    total_to_keep = 0
    total_to_delete = 0

    for cluster_id, indices in enumerate(tqdm(clusters.values(), desc="Building output", unit="cluster")):
        # Find best image
        best_idx = max(indices, key=get_score)
        total_to_keep += 1

        best_res = int(all_resolutions[best_idx])
        best_tags = int(all_tag_counts[best_idx])
        best_size = int(all_file_sizes[best_idx])
        keep_entry = {
            "path": all_paths[best_idx],
            "resolution": best_res,
            "tags": best_tags,
            "size_bytes": best_size,
        }

        delete_entries = []
        for idx in indices:
            if idx != best_idx:
                total_to_delete += 1
                res = int(all_resolutions[idx])
                tags = int(all_tag_counts[idx])
                size = int(all_file_sizes[idx])
                delete_entries.append({
                    "path": all_paths[idx],
                    "resolution": res,
                    "tags": tags,
                    "size_bytes": size,
                })

        clusters_output.append({
            "id": cluster_id + 1,
            "size": len(indices),
            "keep": keep_entry,
            "delete": delete_entries,
        })

    # Sort by cluster size (largest first)
    clusters_output.sort(key=lambda x: x["size"], reverse=True)

    # Build output structure
    output = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "threshold": args.threshold,
            "hash_size": args.hash_size,
            "max_distance": MAX_DISTANCE,
        },
        "summary": {
            "total_images": total_images,
            "unique_clusters": len(clusters),
            "images_in_clusters": total_in_clusters,
            "images_to_keep": total_to_keep,
            "images_to_delete": total_to_delete,
        },
        "clusters": clusters_output,
    }

    # Save clusters JSON
    if save_clusters(output):
        print(f"  Saved to: {get_clusters_path()}")
    else:
        print(f"  WARNING: Failed to save clusters!")
        return False

    # Update progress
    progress["phase"] = 2
    progress["phase2_status"]["clustering_complete"] = True
    save_progress(progress)

    # Cluster size distribution
    print()
    print("Cluster size distribution:")
    sizes = [c["size"] for c in clusters_output]
    size_counts = defaultdict(int)
    for s in sizes:
        if s <= 5:
            size_counts[str(s)] += 1
        elif s <= 10:
            size_counts["6-10"] += 1
        elif s <= 20:
            size_counts["11-20"] += 1
        else:
            size_counts["20+"] += 1

    for label in ["2", "3", "4", "5", "6-10", "11-20", "20+"]:
        if label in size_counts:
            print(f"  {label:>5} images: {size_counts[label]:,} clusters")

    # Summary
    print()
    print("=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Total images scanned:  {total_images:,}")
    print(f"Duplicate clusters:    {len(clusters):,}")
    print(f"Images to KEEP:        {total_to_keep:,}")
    print(f"Images to DELETE:      {total_to_delete:,}")
    print()
    print(f"Review the clusters at: {get_clusters_path()}")
    print()
    print("After reviewing, generate the deletion list:")
    print(f"  python {sys.argv[0]} --generate-deletions --output near_duplicate_deletions.txt")
    print("=" * 70)

    return True


def run_generate_deletions(args):
    """Generate deletion list from reviewed clusters JSON."""
    print("=" * 70)
    print("GENERATING DELETION LIST")
    print("=" * 70)

    # Load clusters
    clusters_data = load_clusters()
    if clusters_data is None:
        print("ERROR: No clusters file found. Run Phase 2 first.")
        return False

    summary = clusters_data.get("summary", {})
    print(f"Clusters found: {summary.get('unique_clusters', 0):,}")
    print(f"Images to delete: {summary.get('images_to_delete', 0):,}")
    print()

    # Extract all deletion paths
    deletion_paths = []
    for cluster in clusters_data.get("clusters", []):
        for entry in cluster.get("delete", []):
            path = entry.get("path")
            if path:
                deletion_paths.append(path)

    # Write to output file
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for path in sorted(deletion_paths):
            f.write(path + '\n')

    print(f"Written {len(deletion_paths):,} paths to: {output_path}")
    print()
    print("To delete the files, you can use:")
    if platform.system() == "Windows":
        print(f'  # Preview (PowerShell): Get-Content "{output_path}" | Select-Object -First 10')
        print(f'  # Delete (PowerShell): Get-Content "{output_path}" | ForEach-Object {{ Remove-Item $_ }}')
    else:
        print(f"  # Preview: head {output_path}")
        print(f"  # Delete: while read -r f; do rm \"$f\"; done < {output_path}")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Find near-duplicate image clusters (scalable version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Scan and hash all shards
  python find_near_dupes_cluster.py --phase 1 --data-root "L:\\Dab\\Dab" --workers 16

  # Resume Phase 1 after interruption
  python find_near_dupes_cluster.py --phase 1 --resume

  # Phase 2: Build clusters and export JSON
  python find_near_dupes_cluster.py --phase 2 --threshold 0.95

  # Generate deletion list after reviewing clusters
  python find_near_dupes_cluster.py --generate-deletions --output near_duplicate_deletions.txt

  # Full run (Phase 1 + Phase 2)
  python find_near_dupes_cluster.py --full --data-root "L:\\Dab\\Dab" --threshold 0.95
"""
    )

    # Phase selection
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        help='Which phase to run: 1=scan/hash, 2=cluster'
    )
    phase_group.add_argument(
        '--full',
        action='store_true',
        help='Run both phases (scan then cluster)'
    )
    phase_group.add_argument(
        '--generate-deletions',
        action='store_true',
        help='Generate deletion list from clusters.json'
    )

    # Data source
    parser.add_argument(
        '--data-root',
        type=str,
        default=r'L:\Dab\Dab',
        help='Root directory containing shard_* folders (default: L:\\Dab\\Dab)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint (Phase 1 only)'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='near_duplicate_deletions.txt',
        help='Output file for deletion list (default: near_duplicate_deletions.txt)'
    )

    # Configuration
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.95,
        help='Similarity threshold 0.0-1.0 (default: 0.95)'
    )
    parser.add_argument(
        '--hash-size',
        type=int,
        default=32,
        help='dHash size (default: 32, produces 1024-bit hash)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=min(16, multiprocessing.cpu_count()),
        help=f'Number of parallel workers (default: {min(16, multiprocessing.cpu_count())})'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress'
    )
    parser.add_argument(
        '--delete-corrupt',
        action='store_true',
        help='Delete images with corrupt EXIF data during Phase 1 (default: disabled)'
    )
    parser.add_argument(
        '--keep-corrupt',
        action='store_true',
        help='Skip (do not delete) images with corrupt EXIF data during Phase 1'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--num-permutations', '-K',
        type=int,
        default=4,
        help='Number of byte permutations for LSH (default: 4, fewer=faster but may miss duplicates)'
    )
    parser.add_argument(
        '--multi-probe',
        action='store_true',
        help='Enable multi-probe LSH (check 1-bit Hamming neighbor buckets)'
    )
    parser.add_argument(
        '--max-bucket-size',
        type=int,
        default=8000,
        help='Max images per bucket before sampling (default: 8000, GPU can handle larger buckets)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU acceleration for Hamming distance (default: enabled if available)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU only)'
    )
    parser.add_argument(
        '--high-recall',
        action='store_true',
        help='Maximize duplicate detection (sets: -K 8 --multi-probe -t 0.92 --max-bucket-size 5000)'
    )
    parser.add_argument(
        '--hash-algorithms',
        type=str,
        default='dhash',
        help='Hash algorithms: dhash, phash, ahash, or comma-separated combo e.g. "dhash,phash" (default: dhash)'
    )
    parser.add_argument(
        '--verify-borderline',
        action='store_true',
        help='Re-verify borderline matches with secondary hash (slower, more accurate)'
    )

    args = parser.parse_args()

    # Configure logging based on --verbose flag
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s: %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )

    # Parse hash algorithms into tuple
    args.hash_algorithms = tuple(a.strip().lower() for a in args.hash_algorithms.split(','))

    # Apply --high-recall preset (override individual settings)
    if args.high_recall:
        args.num_permutations = max(args.num_permutations, 8)
        args.multi_probe = True
        args.threshold = min(args.threshold, 0.92)
        args.max_bucket_size = max(args.max_bucket_size, 12000)  # Larger buckets = less sampling = better recall

    # Validate hash_size parameter
    # Must be multiple of 4 so that hash_size^2 is divisible by 8 (byte-aligned)
    if args.hash_size < 4 or args.hash_size > 32 or args.hash_size % 4 != 0:
        print(f"ERROR: hash_size must be a multiple of 4 between 4 and 32 (got {args.hash_size})")
        return False

    # Validate hash algorithms
    VALID_ALGORITHMS = {'dhash', 'phash', 'ahash'}
    invalid_algos = set(args.hash_algorithms) - VALID_ALGORITHMS
    if invalid_algos:
        print(f"ERROR: Invalid hash algorithm(s): {invalid_algos}. Valid options: {VALID_ALGORITHMS}")
        return False

    # Determine what to run
    if args.generate_deletions:
        return run_generate_deletions(args)
    elif args.phase == 1:
        return run_phase1(args)
    elif args.phase == 2:
        return run_phase2(args)
    elif args.full:
        if run_phase1(args):
            print("\n" + "=" * 70 + "\n")
            return run_phase2(args)
        return False
    else:
        # Default: show help (no action taken)
        parser.print_help()
        print()
        print("TIP: Start with --phase 1 to scan your images, then --phase 2 to build clusters.")
        return None


if __name__ == "__main__":
    success = main()
    # None = no action (help shown), True = success, False = failure
    sys.exit(0 if success is not False else 1)
