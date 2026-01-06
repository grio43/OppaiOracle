from __future__ import annotations
import os
import hmac
import hashlib
import struct
import logging
import uuid
from typing import Optional, Tuple

import torch
from safetensors.torch import save, load, save_file, load_file

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
                return load(data[_LEGACY_DIGEST_SIZE:])["t"]
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
        Path with image extension replaced by sidecar extension

    Raises:
        ValueError: If path contains suspicious patterns like '..'

    Example:
        get_sidecar_path("/data/12345.png") -> "/data/12345.safetensor"
    """
    # Validate path doesn't contain path traversal attempts
    # This prevents writing sidecars outside the intended directory
    if ".." in image_path:
        raise ValueError(f"Path traversal detected in image path: {image_path}")

    base = os.path.splitext(image_path)[0]
    return base + extension


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

        # Convert mask to uint8 for storage efficiency
        if mask_cpu.dtype == torch.bool:
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
        return True

    except Exception as e:
        logging.warning(f"Failed to save sidecar {path}: {e}")
        return False
    finally:
        # Always cleanup temp file if it exists (handles both success and failure cases)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logging.debug(f"Failed to cleanup temp file {tmp_path}: {cleanup_err}")


def load_sidecar(
    path: str,
    expected_config_hash: Optional[str] = None,
    expected_source_mtime: Optional[float] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load preprocessed image and mask from a sidecar file.

    Args:
        path: Sidecar file path
        expected_config_hash: If provided, validates against stored hash.
            Returns None if hash mismatch (cache invalidation).
        expected_source_mtime: If provided, validates against stored source mtime.
            Returns None if source file has been modified since caching.

    Returns:
        Tuple of (image, mask) tensors if successful and validation passes,
        None if file doesn't exist, is invalid, or validation fails.

    Note:
        Mask is returned as torch.bool dtype.
    """
    # Removed os.path.exists() check to prevent TOCTOU race condition
    # Another worker could delete/replace file between check and open
    # Instead, handle FileNotFoundError directly in the exception handler

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
                if stored_mtime_str is not None:
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
            image = f.get_tensor("image")
            mask = f.get_tensor("mask")

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
            unique_vals = mask.unique()
            if len(unique_vals) > 0:
                max_val = unique_vals.max().item()
                if max_val > 1:
                    logging.warning(
                        f"Sidecar {path} mask contains invalid values (max={max_val}), "
                        "expected 0 or 1 only. Mask may be corrupted."
                    )
                    return None
            mask = mask.to(torch.bool)

        return image, mask

    except FileNotFoundError:
        # Normal cache miss - file doesn't exist
        return None
    except Exception as e:
        logging.warning(f"Failed to load sidecar {path}: {e}")
        return None


__all__ = ["encode_tensor", "decode_tensor", "get_sidecar_path", "save_sidecar", "load_sidecar"]
