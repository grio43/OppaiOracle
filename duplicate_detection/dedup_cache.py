"""
Deduplication Hash Cache Utilities

Provides persistent storage for perceptual hashes using safetensors format.
Each shard gets its own cache file for resumable processing.
"""

from __future__ import annotations
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Lazy import torch to avoid startup overhead
_torch = None
_safetensors = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_safetensors():
    global _safetensors
    if _safetensors is None:
        from safetensors.torch import save_file, load_file
        from safetensors import safe_open
        _safetensors = {'save_file': save_file, 'load_file': load_file, 'safe_open': safe_open}
    return _safetensors


# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "logs" / "dedup_hashes"
PROGRESS_FILE = "dedup_progress.json"
CLUSTERS_FILE = "dedup_clusters.json"


@dataclass
class ShardHashData:
    """Container for shard hash cache data."""
    shard_name: str
    paths: List[str]  # Relative paths within shard
    hashes: np.ndarray  # N x hash_byte_count bytes (hash_size^2 * num_algorithms bits as uint8)
    resolutions: np.ndarray  # N x 1 int64
    tag_counts: np.ndarray  # N x 1 int32
    file_sizes: np.ndarray  # N x 1 int64
    hash_size: int = 16
    hash_algorithms: tuple = ('dhash',)  # Which algorithms were used
    created_at: str = ""

    def __len__(self) -> int:
        return len(self.paths)

    def get_hash_int(self, idx: int) -> int:
        """Get hash as integer for comparison."""
        h_bytes = self.hashes[idx]
        return int.from_bytes(h_bytes.tobytes(), 'big')


def get_cache_dir() -> Path:
    """Get the dedup cache directory, creating if needed."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_hash_cache_path(shard_name: str, cache_dir: Optional[Path] = None) -> Path:
    """Get path for shard hash cache file."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    return cache_dir / f"{shard_name}.hashcache.safetensor"


def get_progress_path(cache_dir: Optional[Path] = None) -> Path:
    """Get path for progress tracking file."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    return cache_dir / PROGRESS_FILE


def get_clusters_path(cache_dir: Optional[Path] = None) -> Path:
    """Get path for clusters output file."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    return cache_dir / CLUSTERS_FILE


def save_shard_hashes(
    shard_name: str,
    paths: List[str],
    hashes: np.ndarray,
    resolutions: np.ndarray,
    tag_counts: np.ndarray,
    file_sizes: np.ndarray,
    hash_size: int = 16,
    hash_algorithms: tuple = ('dhash',),
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Save computed hashes for a shard to persistent storage.

    Args:
        shard_name: Name of the shard (e.g., 'shard_00000')
        paths: List of image paths (relative to shard root)
        hashes: N x hash_byte_count byte array (hash_size^2 * num_algorithms bits)
        resolutions: N element array of width * height values
        tag_counts: N element array of tag counts
        file_sizes: N element array of file sizes in bytes
        hash_size: Hash size per algorithm (default 16 for 256-bit each)
        hash_algorithms: Tuple of algorithm names used (default: ('dhash',))
        cache_dir: Optional override for cache directory

    Returns:
        True if saved successfully, False on error
    """
    torch = _get_torch()
    sf = _get_safetensors()

    cache_path = get_hash_cache_path(shard_name, cache_dir)
    tmp_path = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")

    try:
        n = len(paths)
        if n == 0:
            logging.warning(f"No images to cache for shard {shard_name}")
            return False

        # Validate array shapes - hash_byte_count depends on hash_size and num algorithms
        num_algorithms = len(hash_algorithms)
        hash_byte_count = ((hash_size * hash_size) // 8) * num_algorithms
        assert hashes.shape == (n, hash_byte_count), f"Hashes shape mismatch: {hashes.shape} != ({n}, {hash_byte_count})"
        assert len(resolutions) == n, f"Resolutions length mismatch: {len(resolutions)} != {n}"
        assert len(tag_counts) == n, f"Tag counts length mismatch: {len(tag_counts)} != {n}"
        assert len(file_sizes) == n, f"File sizes length mismatch: {len(file_sizes)} != {n}"

        # Convert to tensors
        tensors = {
            "hashes": torch.from_numpy(hashes.astype(np.uint8)),
            "resolutions": torch.from_numpy(resolutions.astype(np.int64)),
            "tag_counts": torch.from_numpy(tag_counts.astype(np.int32)),
            "file_sizes": torch.from_numpy(file_sizes.astype(np.int64)),
        }

        # Store paths in metadata (JSON-encoded to handle special characters)
        metadata = {
            "shard_name": shard_name,
            "image_count": str(n),
            "hash_size": str(hash_size),
            "hash_algorithms": json.dumps(hash_algorithms),
            "version": "1.2",
            "created_at": datetime.now().isoformat(),
            "paths": json.dumps(paths),
        }

        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        sf['save_file'](tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, cache_path)

        logging.debug(f"Saved {n} hashes for {shard_name}")
        return True

    except Exception as e:
        logging.error(f"Failed to save shard hashes for {shard_name}: {e}")
        return False
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def load_shard_hashes(
    shard_name: str,
    cache_dir: Optional[Path] = None,
) -> Optional[ShardHashData]:
    """
    Load cached hashes for a shard.

    Args:
        shard_name: Name of the shard (e.g., 'shard_00000')
        cache_dir: Optional override for cache directory

    Returns:
        ShardHashData if cache exists and is valid, None otherwise
    """
    sf = _get_safetensors()

    cache_path = get_hash_cache_path(shard_name, cache_dir)

    try:
        with sf['safe_open'](str(cache_path), framework="pt") as f:
            metadata = f.metadata()

            if not metadata:
                logging.warning(f"Cache {cache_path} has no metadata")
                return None

            # Load tensors
            hashes = f.get_tensor("hashes").numpy()
            resolutions = f.get_tensor("resolutions").numpy()
            tag_counts = f.get_tensor("tag_counts").numpy()
            file_sizes = f.get_tensor("file_sizes").numpy()

            # Parse paths from metadata (handle both JSON and legacy newline format)
            paths_str = metadata.get("paths", "")
            if paths_str.startswith("["):
                # New JSON format (version 1.1+)
                paths = json.loads(paths_str)
            else:
                # Legacy newline-separated format (version 1.0)
                paths = paths_str.split("\n") if paths_str else []

            # Validate counts match
            n = int(metadata.get("image_count", "0"))
            if len(paths) != n:
                logging.warning(f"Path count mismatch in {cache_path}: {len(paths)} != {n}")
                return None
            if len(hashes) != n:
                logging.warning(f"Hash count mismatch in {cache_path}: {len(hashes)} != {n}")
                return None

            # Parse hash_algorithms from metadata (default to dhash for backward compat)
            hash_algorithms_str = metadata.get("hash_algorithms", '["dhash"]')
            try:
                hash_algorithms = tuple(json.loads(hash_algorithms_str))
            except (json.JSONDecodeError, TypeError):
                hash_algorithms = ('dhash',)

            return ShardHashData(
                shard_name=shard_name,
                paths=paths,
                hashes=hashes,
                resolutions=resolutions,
                tag_counts=tag_counts.astype(np.int32),
                file_sizes=file_sizes,
                hash_size=int(metadata.get("hash_size", "16")),
                hash_algorithms=hash_algorithms,
                created_at=metadata.get("created_at", ""),
            )

    except FileNotFoundError:
        return None
    except Exception as e:
        logging.warning(f"Failed to load shard hashes for {shard_name}: {e}")
        return None


def load_progress(cache_dir: Optional[Path] = None) -> Dict:
    """Load progress tracking state."""
    progress_path = get_progress_path(cache_dir)

    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "version": "1.0",
            "phase": 0,
            "phase1_status": {
                "total_shards": 0,
                "completed_shards": [],
                "failed_shards": [],
                "started_at": None,
                "last_updated": None,
            },
            "phase2_status": {
                "started": False,
                "lsh_complete": False,
                "clustering_complete": False,
            },
            "config": {},
        }
    except Exception as e:
        logging.error(f"Failed to load progress: {e}")
        # Return default structure instead of empty dict to prevent KeyError
        return {
            "version": "1.0",
            "phase": 0,
            "phase1_status": {
                "total_shards": 0,
                "completed_shards": [],
                "failed_shards": [],
                "started_at": None,
                "last_updated": None,
            },
            "phase2_status": {
                "started": False,
                "lsh_complete": False,
                "clustering_complete": False,
            },
            "config": {},
        }


def save_progress(progress: Dict, cache_dir: Optional[Path] = None) -> bool:
    """Save progress tracking state."""
    progress_path = get_progress_path(cache_dir)
    tmp_path = progress_path.with_suffix(f".{uuid.uuid4().hex}.tmp")

    try:
        # Defensive check: ensure phase1_status exists before accessing
        if "phase1_status" not in progress:
            progress["phase1_status"] = {}
        progress["phase1_status"]["last_updated"] = datetime.now().isoformat()

        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)

        os.replace(tmp_path, progress_path)
        return True

    except Exception as e:
        logging.error(f"Failed to save progress: {e}")
        return False
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def save_clusters(
    clusters_data: Dict,
    cache_dir: Optional[Path] = None,
) -> bool:
    """Save cluster analysis results to JSON."""
    clusters_path = get_clusters_path(cache_dir)
    tmp_path = clusters_path.with_suffix(f".{uuid.uuid4().hex}.tmp")

    try:
        clusters_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, indent=2)

        os.replace(tmp_path, clusters_path)
        return True

    except Exception as e:
        logging.error(f"Failed to save clusters: {e}")
        return False
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def load_clusters(cache_dir: Optional[Path] = None) -> Optional[Dict]:
    """Load cluster analysis results from JSON."""
    clusters_path = get_clusters_path(cache_dir)

    try:
        with open(clusters_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.error(f"Failed to load clusters: {e}")
        return None


def iter_cached_shards(cache_dir: Optional[Path] = None):
    """
    Iterate over all cached shard hash files.

    Yields:
        ShardHashData for each valid cache file
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    for cache_file in sorted(cache_dir.glob("*.hashcache.safetensor")):
        shard_name = cache_file.stem.replace(".hashcache", "")
        data = load_shard_hashes(shard_name, cache_dir)
        if data is not None:
            yield data


def iter_cached_shard_metadata(cache_dir: Optional[Path] = None):
    """
    Iterate over cached shards, yielding only metadata (no tensor loading).

    This is much faster than iter_cached_shards() when you only need
    shard names and image counts for pre-allocation.

    Yields:
        Tuple of (shard_name, image_count, hash_size, hash_algorithms)
    """
    sf = _get_safetensors()
    if cache_dir is None:
        cache_dir = get_cache_dir()

    for cache_file in sorted(cache_dir.glob("*.hashcache.safetensor")):
        shard_name = cache_file.stem.replace(".hashcache", "")
        try:
            with sf['safe_open'](str(cache_file), framework="pt") as f:
                metadata = f.metadata()
                if metadata:
                    # Parse hash_algorithms from metadata
                    hash_algorithms_str = metadata.get("hash_algorithms", '["dhash"]')
                    try:
                        hash_algorithms = tuple(json.loads(hash_algorithms_str))
                    except (json.JSONDecodeError, TypeError):
                        hash_algorithms = ('dhash',)

                    yield (
                        shard_name,
                        int(metadata.get("image_count", "0")),
                        int(metadata.get("hash_size", "16")),
                        hash_algorithms,
                    )
        except Exception as e:
            logging.warning(f"Failed to read metadata for {shard_name}: {e}")


def get_cached_shard_count(cache_dir: Optional[Path] = None) -> int:
    """Count number of cached shard files."""
    if cache_dir is None:
        cache_dir = get_cache_dir()

    return len(list(cache_dir.glob("*.hashcache.safetensor")))


__all__ = [
    'ShardHashData',
    'get_cache_dir',
    'get_hash_cache_path',
    'get_progress_path',
    'get_clusters_path',
    'save_shard_hashes',
    'load_shard_hashes',
    'load_progress',
    'save_progress',
    'save_clusters',
    'load_clusters',
    'iter_cached_shards',
    'iter_cached_shard_metadata',
    'get_cached_shard_count',
]
