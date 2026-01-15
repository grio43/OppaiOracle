from __future__ import annotations
import logging
import os
import hashlib
from typing import Sequence, Tuple

_logger = logging.getLogger(__name__)

def compute_cache_config_hash(
    *,
    image_size: int,
    pad_color: Sequence[int] | Tuple[int, int, int],
    normalize_mean: Sequence[float] | Tuple[float, float, float],
    normalize_std: Sequence[float] | Tuple[float, float, float],
    storage_dtype: str,
    vocab_size: int | None = None,
    has_joint_transforms: bool = False,
    random_flip_prob: float = 0.0,
    has_orientation_handler: bool = False,
) -> str:
    """
    Build a short, stable hash of preprocessing settings used to key sidecar cache entries.

    Use CACHE_SCHEMA_VERSION env var to intentionally bust keys when semantics change.

    Args:
        image_size: Target image size for preprocessing.
        pad_color: RGB padding color tuple.
        normalize_mean: Normalization mean per channel.
        normalize_std: Normalization std per channel.
        storage_dtype: Storage dtype for sidecar cache (e.g., "bfloat16").
        vocab_size: Number of tags in vocabulary. When provided, cache keys are
            invalidated if vocabulary size changes (e.g., tags added/removed).
            This prevents dimension mismatches in cached tag vectors.
        has_joint_transforms: Whether joint transforms are applied. When True,
            cache is invalidated since transforms affect output tensors.
        random_flip_prob: Probability of random horizontal flip. Included to ensure
            cache invalidation when flip augmentation settings change.
        has_orientation_handler: Whether orientation handler is active for tag swapping
            during flips. Included to ensure cache invalidation when flip behavior changes.

    Returns an 8-character SHA256 prefix.
    """
    try:
        schema_version = os.getenv("CACHE_SCHEMA_VERSION", "v2")  # Keep v2 - flip params don't affect cached data
        cfg_fields = {
            "image_size": int(image_size),
            "pad_color": tuple(pad_color),
            "normalize_mean": tuple(normalize_mean),
            "normalize_std": tuple(normalize_std),
            # Keep as "l2_storage_dtype" for hash backward compatibility
            "l2_storage_dtype": str(storage_dtype),
            "schema_version": schema_version,
            "has_joint_transforms": bool(has_joint_transforms),
            # NOTE: random_flip_prob, has_orientation_handler, and vocab_size are NOT included in hash
            # because cached data is always the canonical UNFLIPPED image/mask and is independent 
            # of the tag vocabulary. Including them would cause unnecessary cache invalidation.
        }
        # Sort keys for deterministic hash ordering across Python versions
        cfg_str = "|".join(f"{k}={v}" for k, v in sorted(cfg_fields.items()))
        return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
    except Exception as e:
        # Log error instead of silent fallback - configuration errors should be visible
        _logger.error(
            f"Failed to compute cache config hash: {e}. "
            "This indicates a configuration error that will cause cache misses."
        )
        raise ValueError(f"Invalid cache configuration: {e}") from e


# Backward compatibility alias
compute_l2_cfg_hash = compute_cache_config_hash

__all__ = ["compute_cache_config_hash", "compute_l2_cfg_hash"]
