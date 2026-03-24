from __future__ import annotations
import os
import hmac
import hashlib
import struct
import logging
import uuid
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import torch
from safetensors.torch import save, load, save_file, load_file


# =============================================================================
# LRU Cache for Sidecar Tensors
# =============================================================================
# Performance optimization: Avoid repeated tensor cloning on cache hits.
# Memory-mapped tensors from safetensors must be cloned to avoid Windows file
# locking issues. This LRU cache stores already-cloned tensors, eliminating
# the clone overhead (~512KB memcpy per image) for recently accessed items.
#
# Thread-safe implementation using a lock since DataLoader workers may access
# concurrently in threaded mode (num_workers=0 with threads or shared cache).
# =============================================================================

class _SidecarLRUCache:
    """
    Thread-safe LRU cache for cloned sidecar tensors.

    Caches (image, mask) tensor pairs by (path, config_hash) to avoid repeated
    cloning of memory-mapped tensors. Returns deep clones from cache to ensure
    callers can safely modify tensors without affecting cached copies.
    """

    def __init__(self, max_size: int = 256):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of (image, mask) pairs to cache.
                      Default 256 entries ~= 128MB for 512x512 RGB images.
        """
        self._cache: OrderedDict[Tuple[str, Optional[str]], Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Tuple[str, Optional[str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached tensors, returning clones to prevent modification of cached data.

        Args:
            key: Tuple of (sidecar_path, config_hash)

        Returns:
            Tuple of (image, mask) tensor clones if cached, None otherwise.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            image, mask = self._cache[key]
            self._hits += 1

            # Return clones so callers can safely modify without affecting cache
            # Use non_blocking=False to ensure clone completes before return
            return image.clone(), mask.clone()

    def put(self, key: Tuple[str, Optional[str]], image: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Cache tensor pair. Tensors should already be cloned from mmap.

        Args:
            key: Tuple of (sidecar_path, config_hash)
            image: Image tensor (already cloned from mmap)
            mask: Mask tensor (already cloned from mmap)
        """
        with self._lock:
            # Remove if exists to update position
            if key in self._cache:
                del self._cache[key]

            # Add to end (most recently used)
            self._cache[key] = (image, mask)

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, path: str) -> None:
        """
        Invalidate all cache entries for a given path (any config_hash).

        Call this when a sidecar file is updated/replaced.

        Args:
            path: Sidecar file path to invalidate
        """
        with self._lock:
            # Find and remove all entries for this path
            keys_to_remove = [k for k in self._cache if k[0] == path]
            for k in keys_to_remove:
                del self._cache[k]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# Global sidecar cache instance
# Size can be adjusted via set_sidecar_cache_size() before training starts
_sidecar_cache = _SidecarLRUCache(max_size=256)


def set_sidecar_cache_size(max_size: int) -> None:
    """
    Set the maximum size of the sidecar LRU cache.

    Call this before training starts to adjust cache size based on available
    memory and dataset access patterns. Clears existing cache.

    Args:
        max_size: Maximum number of (image, mask) pairs to cache.
                  Set to 0 to disable caching.
    """
    global _sidecar_cache
    _sidecar_cache = _SidecarLRUCache(max_size=max_size)


def get_sidecar_cache_stats() -> dict:
    """Return sidecar cache statistics for monitoring."""
    return _sidecar_cache.stats()


def clear_sidecar_cache() -> None:
    """Clear the sidecar tensor cache."""
    _sidecar_cache.clear()

# Optional HMAC key for integrity checks
_HMAC_KEY = os.environ.get("CACHE_CODEC_HMAC_KEY")

# Supported HMAC algorithms with their digest sizes (algorithm agility)
_HMAC_ALGORITHMS = {
    b'\x01': ('sha256', 32),
    b'\x02': ('sha384', 48),
    b'\x03': ('sha512', 64),
}
_DEFAULT_HMAC_ALG = b'\x01'  # sha256
_HMAC_ALG_ENV = os.environ.get("CACHE_CODEC_HMAC_ALG", "sha256")

# Legacy digest size for backward compatibility
_LEGACY_DIGEST_SIZE = 32  # sha256

# Format markers for codec
# SECURITY: Removed pickle format (b'P') to prevent arbitrary code execution
# Only safetensors format is now supported for new writes
_FORMAT_SAFETENSORS = b'S'
_FORMAT_LEGACY_PICKLE = b'P'  # Read-only for migration, logged as warning


def _resolve_key(key: Optional[bytes | str]) -> Optional[bytes]:
    if key is None:
        key = _HMAC_KEY
    if key is None:
        return None
    if isinstance(key, str):
        return key.encode("utf-8")
    return key


def _get_hmac_alg_id() -> bytes:
    """Get the HMAC algorithm ID based on environment configuration."""
    alg_name = _HMAC_ALG_ENV.lower()
    for alg_id, (name, _) in _HMAC_ALGORITHMS.items():
        if name == alg_name:
            return alg_id
    return _DEFAULT_HMAC_ALG


def _get_hmac_info(alg_id: bytes) -> Tuple[str, int]:
    """Get algorithm name and digest size for an algorithm ID."""
    if alg_id in _HMAC_ALGORITHMS:
        return _HMAC_ALGORITHMS[alg_id]
    # Default to sha256 for unknown algorithms
    return ('sha256', 32)


def encode_tensor(t: torch.Tensor, key: Optional[bytes | str] = None) -> bytes:
    """
    Serialize a tensor to bytes using safetensors with optional HMAC.

    SECURITY: Only safetensors format is used. Pickle has been removed to
    prevent arbitrary code execution vulnerabilities.

    The tensor is detached and moved to CPU before encoding to avoid
    capturing device-specific state. When a key is provided, an HMAC
    digest of the payload is prepended with algorithm identifier.

    Format (with HMAC):
    - [alg_id (1 byte)] + [HMAC digest (variable)] + [format marker (1 byte)] + [payload]

    Format (without HMAC):
    - [format marker (1 byte)] + [payload]
    """
    t_cpu = t.detach().cpu().contiguous()

    # Always use safetensors for security
    # save() returns bytes directly - no need for BytesIO buffer
    payload = save({"t": t_cpu})
    data = _FORMAT_SAFETENSORS + payload

    # Add HMAC if key provided (with algorithm agility)
    k = _resolve_key(key)
    if k:
        alg_id = _get_hmac_alg_id()
        alg_name, _ = _get_hmac_info(alg_id)
        digest = hmac.new(k, data, alg_name).digest()
        data = alg_id + digest + data
    return data


def decode_tensor(b: bytes, key: Optional[bytes | str] = None) -> torch.Tensor:
    """
    Deserialize bytes back into a tensor, verifying HMAC if present.

    Supports safetensors format only. Legacy pickle format has been permanently
    disabled for security reasons (prevents arbitrary code execution).

    SECURITY: Pickle deserialization is permanently disabled. If you have old
    pickle-format cache files, delete them and re-cache your data.

    With a key, the payload must include algorithm ID + digest; the digest
    is recomputed and compared using compare_digest() to mitigate timing
    attacks.
    """
    k = _resolve_key(key)
    data = b

    # HMAC verification if key provided
    if k:
        if len(data) < 2:  # Need at least alg_id + some data
            raise ValueError("Payload too short for HMAC")

        # Check for new format with algorithm ID
        alg_id = data[0:1]
        if alg_id in _HMAC_ALGORITHMS:
            # New format: [alg_id (1)] + [digest (N)] + [payload]
            alg_name, digest_size = _get_hmac_info(alg_id)
            if len(data) < 1 + digest_size:
                raise ValueError(f"Payload too short for {alg_name} HMAC digest")
            digest = data[1:1 + digest_size]
            payload = data[1 + digest_size:]
            expected = hmac.new(k, payload, alg_name).digest()
            if not hmac.compare_digest(digest, expected):
                raise ValueError("HMAC verification failed")
            data = payload
        else:
            # Legacy format: [digest (32)] + [payload] (sha256 assumed)
            if len(data) < _LEGACY_DIGEST_SIZE:
                raise ValueError("Payload too short for legacy HMAC digest")
            digest = data[:_LEGACY_DIGEST_SIZE]
            payload = data[_LEGACY_DIGEST_SIZE:]
            expected = hmac.new(k, payload, "sha256").digest()
            if not hmac.compare_digest(digest, expected):
                raise ValueError("HMAC verification failed")
            data = payload

    # Check format marker and decode accordingly
    if len(data) > 0:
        format_marker = data[0:1]

        if format_marker == _FORMAT_LEGACY_PICKLE:
            # SECURITY: Pickle has been permanently disabled to prevent arbitrary code execution
            # See: https://docs.python.org/3/library/pickle.html#restricting-globals
            raise ValueError(
                "Pickle format cache entry detected. Pickle deserialization has been permanently "
                "disabled for security (prevents arbitrary code execution). Please delete the old "
                "cache files and re-cache your data using the current safetensors format."
            )

        elif format_marker == _FORMAT_SAFETENSORS:
            # Safetensors format with marker (current format)
            return load(data[1:])["t"]

    # Legacy format fallback: try direct safetensors load (old format without marker)
    try:
        return load(data)["t"]
    except (RuntimeError, KeyError) as e:
        # safetensors.load raises RuntimeError for invalid format
        # KeyError if "t" key missing
        # Only try digest-skip fallback if NO HMAC key was provided.
        # If HMAC key was provided, digest was already stripped above.
        if not k and len(data) > _LEGACY_DIGEST_SIZE:
            try:
                result = load(data[_LEGACY_DIGEST_SIZE:])["t"]
                logging.warning(
                    "Loaded tensor using legacy digest-skip fallback format. "
                    "Consider re-caching data for reliability."
                )
                return result
            except (RuntimeError, KeyError):
                pass  # Fallback failed
        # Re-raise original error
        raise ValueError(f"Failed to decode cache entry: {e}") from e


def get_sidecar_path(image_path: str, extension: str = ".safetensor") -> str:
    """
    Compute the sidecar cache file path for an image.

    Args:
        image_path: Path to the original image file
        extension: Sidecar file extension (default: .safetensor)

    Returns:
        Path with sidecar extension appended to the original filename.
        Using the full filename (e.g., image.jpg.safetensor) avoids collisions 
        between different formats of the same image (e.g., image.jpg and image.png).

    Raises:
        ValueError: If path contains suspicious patterns like '..'
    """
    # Validate path doesn't contain path traversal attempts
    if ".." in image_path:
        raise ValueError(f"Path traversal detected in image path: {image_path}")

    # Append extension to full path to avoid stem-based collisions
    return image_path + extension


def save_sidecar(
    path: str,
    image: torch.Tensor,
    mask: torch.Tensor,
    config_hash: str,
    image_size: Optional[int] = None,
    source_mtime: Optional[float] = None,
) -> bool:
    """
    Save preprocessed image and mask to a sidecar file.

    Args:
        path: Output sidecar file path
        image: Preprocessed image tensor (3, H, W), any dtype
        mask: Padding mask tensor (H, W), bool or uint8
        config_hash: Hash of preprocessing config for validation
        image_size: Optional image size to store in metadata
        source_mtime: Optional source file modification time for cache invalidation

    Returns:
        True if saved successfully, False on error

    File format:
        Safetensors file with:
        - "image": (3, H, W) tensor
        - "mask": (H, W) uint8 tensor
        - metadata: {"config_hash": "...", "image_size": "...", "source_mtime": "..."}
    """
    # Use UUID in temp filename to avoid race conditions between workers
    # PID is not sufficient because DataLoader workers share the same PID in spawn context
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        # Validate input tensor shapes
        if image.dim() != 3:
            raise ValueError(f"Image must be 3D (C, H, W), got {image.dim()}D with shape {tuple(image.shape)}")
        if image.shape[0] != 3:
            raise ValueError(f"Image must have 3 channels, got {image.shape[0]}")
        if mask.dim() != 2:
            raise ValueError(f"Mask must be 2D (H, W), got {mask.dim()}D with shape {tuple(mask.shape)}")
        if image.shape[1:] != mask.shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} must match image spatial dims {tuple(image.shape[1:])}")

        # Prepare tensors
        img_cpu = image.detach().cpu().contiguous()
        mask_cpu = mask.detach().cpu().contiguous()

        # Convert mask to uint8 for storage efficiency and consistency
        # Handle any input dtype by converting through bool first
        if mask_cpu.dtype != torch.uint8:
            if mask_cpu.dtype != torch.bool:
                # Convert non-bool dtypes (float, int32, etc.) to bool first
                mask_cpu = mask_cpu.to(torch.bool)
            mask_cpu = mask_cpu.to(torch.uint8)

        tensors = {
            "image": img_cpu,
            "mask": mask_cpu,
        }

        # Build metadata
        metadata = {"config_hash": config_hash}
        if image_size is not None:
            metadata["image_size"] = str(image_size)
        if source_mtime is not None:
            metadata["source_mtime"] = str(source_mtime)

        # Ensure parent directory exists
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Atomic write: write to temp file then rename
        save_file(tensors, tmp_path, metadata=metadata)

        # Atomic rename (works on same filesystem)
        os.replace(tmp_path, path)

        # Invalidate LRU cache entry for this path since file contents changed
        _sidecar_cache.invalidate(path)

        return True

    except Exception as e:
        logging.warning(f"Failed to save sidecar {path}: {e}")
        return False
    finally:
        # Always cleanup temp file if it exists (handles both success and failure cases)
        # Use try/except instead of exists() check to avoid TOCTOU race condition
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass  # Already cleaned up (e.g., after successful rename)
        except Exception as cleanup_err:
            logging.debug(f"Failed to cleanup temp file {tmp_path}: {cleanup_err}")


def load_sidecar(
    path: str,
    expected_config_hash: Optional[str] = None,
    expected_source_mtime: Optional[float] = None,
    use_lru_cache: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load preprocessed image and mask from a sidecar file.

    Args:
        path: Sidecar file path
        expected_config_hash: If provided, validates against stored hash.
            Returns None if hash mismatch (cache invalidation).
        expected_source_mtime: If provided, validates against stored source mtime.
            Returns None if source file has been modified since caching.
        use_lru_cache: If True (default), use in-memory LRU cache to avoid
            repeated tensor cloning for recently accessed files.

    Returns:
        Tuple of (image, mask) tensors if successful and validation passes,
        None if file doesn't exist, is invalid, or validation fails.

    Note:
        Mask is returned as torch.bool dtype.

    Performance:
        When use_lru_cache=True, recently accessed tensors are served from an
        in-memory cache, avoiding the ~512KB memcpy overhead of cloning
        memory-mapped tensors. The cache returns clones to ensure callers
        can safely modify tensors without affecting cached copies.
    """
    # Removed os.path.exists() check to prevent TOCTOU race condition
    # Another worker could delete/replace file between check and open
    # Instead, handle FileNotFoundError directly in the exception handler

    # ===========================================================================
    # LRU Cache Lookup (Performance Optimization)
    # ===========================================================================
    # Check in-memory cache first to avoid disk I/O and tensor cloning overhead.
    # Cache key includes config_hash to handle config changes correctly.
    # Note: We don't include expected_source_mtime in key because:
    # 1. It's rarely used (skipped for performance in main training loop)
    # 2. If mtime validation is needed, we should read from disk anyway
    # ===========================================================================
    cache_key = (path, expected_config_hash)

    if use_lru_cache and expected_source_mtime is None:
        cached = _sidecar_cache.get(cache_key)
        if cached is not None:
            # Cache hit! Return clones (get() already clones for safety)
            return cached

    try:
        # Load with metadata
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            # Check config hash if provided
            metadata = f.metadata()

            # Consistent null safety: always check metadata before accessing
            if expected_config_hash is not None:
                if not metadata:
                    logging.debug(f"Sidecar {path} has no metadata, cannot verify config hash")
                    return None
                stored_hash = metadata.get("config_hash")
                if stored_hash != expected_config_hash:
                    logging.debug(
                        f"Sidecar config hash mismatch: {stored_hash} != {expected_config_hash}"
                    )
                    return None

            # Check source file mtime if provided (cache invalidation when source modified)
            if expected_source_mtime is not None:
                if not metadata:
                    logging.debug(f"Sidecar {path} has no metadata, cannot verify source mtime")
                    return None
                stored_mtime_str = metadata.get("source_mtime")
                if stored_mtime_str is None:
                    # Mtime validation requested but cache entry has no stored mtime
                    # Treat as cache miss to ensure fresh data
                    logging.debug(f"Sidecar {path} missing source_mtime, cannot verify")
                    return None
                try:
                    stored_mtime = float(stored_mtime_str)
                    # Allow small tolerance for floating point comparison (0.001 seconds)
                    if abs(stored_mtime - expected_source_mtime) > 0.001:
                        logging.debug(
                            f"Sidecar source mtime mismatch: {stored_mtime} != {expected_source_mtime}"
                        )
                        return None
                except (ValueError, TypeError):
                    # Invalid stored mtime, treat as cache miss
                    logging.debug(f"Sidecar has invalid source_mtime: {stored_mtime_str}")
                    return None

            # Validate required tensors exist before loading
            keys = set(f.keys())
            if "image" not in keys:
                logging.warning(f"Sidecar {path} missing 'image' tensor (found: {keys})")
                return None
            if "mask" not in keys:
                logging.warning(f"Sidecar {path} missing 'mask' tensor (found: {keys})")
                return None

            # Load tensors
            # CLONE tensors to ensure they don't hold a reference to the memory-mapped file.
            # This is CRITICAL on Windows to allow the file to be replaced/deleted by other
            # workers while training is running. Without clone(), the mmap remains active
            # as long as the tensor is in memory, locking the file.
            image = f.get_tensor("image").clone()
            mask = f.get_tensor("mask").clone()

        # Validate image is 3D (C, H, W)
        if image.dim() != 3:
            logging.warning(f"Sidecar image must be 3D, got {image.dim()}D with shape {tuple(image.shape)}")
            return None

        # Validate image tensor doesn't contain NaN or Inf values (corrupted cache)
        if torch.isnan(image).any():
            logging.warning(f"Sidecar {path} contains NaN values in image tensor, treating as cache miss")
            return None
        if torch.isinf(image).any():
            logging.warning(f"Sidecar {path} contains Inf values in image tensor, treating as cache miss")
            return None

        # Validate mask is 2D (H, W)
        if mask.dim() != 2:
            logging.warning(f"Sidecar mask must be 2D, got {mask.dim()}D with shape {tuple(mask.shape)}")
            return None

        # Validate mask shape matches image spatial dimensions
        if image.shape[-2:] != mask.shape:
            logging.warning(
                f"Sidecar mask shape {tuple(mask.shape)} != image spatial dims {tuple(image.shape[-2:])}"
            )
            return None

        # Convert mask back to bool with validation
        if mask.dtype == torch.uint8:
            # Validate uint8 values are 0 or 1 to prevent silent corruption
            # Values 2-255 would silently become True, corrupting the mask
            # Use efficient check instead of .unique() for large masks
            if (mask > 1).any():
                logging.warning(
                    f"Sidecar {path} mask contains invalid values (>1), "
                    "expected 0 or 1 only. Mask may be corrupted."
                )
                return None
            mask = mask.to(torch.bool)
        elif mask.dtype != torch.bool:
            # Handle unexpected dtypes (e.g., old cache with float/int masks)
            logging.debug(f"Sidecar mask has dtype {mask.dtype}, converting to bool")
            mask = mask.to(torch.bool)

        # ===========================================================================
        # Store in LRU Cache (Performance Optimization)
        # ===========================================================================
        # Cache the validated, cloned tensors for future access. The tensors are
        # already cloned from the mmap above, so we store them directly.
        # On subsequent accesses, _sidecar_cache.get() will return fresh clones
        # to ensure callers can safely modify without affecting the cached copy.
        # ===========================================================================
        if use_lru_cache and expected_source_mtime is None:
            _sidecar_cache.put(cache_key, image, mask)

        return image, mask

    except FileNotFoundError:
        # Normal cache miss - file doesn't exist
        return None
    except Exception as e:
        logging.warning(f"Failed to load sidecar {path}: {e}")
        return None


__all__ = [
    "encode_tensor",
    "decode_tensor",
    "get_sidecar_path",
    "save_sidecar",
    "load_sidecar",
    # LRU cache management functions
    "set_sidecar_cache_size",
    "get_sidecar_cache_stats",
    "clear_sidecar_cache",
]
