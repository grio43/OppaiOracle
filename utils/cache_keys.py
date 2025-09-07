from __future__ import annotations
import os
import hashlib
from typing import Sequence, Tuple

def compute_l2_cfg_hash(
    *,
    image_size: int,
    pad_color: Sequence[int] | Tuple[int, int, int],
    normalize_mean: Sequence[float] | Tuple[float, float, float],
    normalize_std: Sequence[float] | Tuple[float, float, float],
    l2_dtype_str: str,
) -> str:
    """
    Build a short, stable hash of L2-relevant preprocessing settings used to key L2 entries.

    Option B: Exclude any L1 (pre-normalization cache) dtype from the hash so that changing
    in-memory or L1 cache precision does not invalidate or cause false misses in the L2 cache.
    Use CACHE_SCHEMA_VERSION to intentionally bust keys when semantics change.

    Returns an 8-character SHA256 prefix.
    """
    try:
        schema_version = os.getenv("CACHE_SCHEMA_VERSION", "v1")
        cfg_fields = {
            "image_size": int(image_size),
            "pad_color": tuple(pad_color),
            "normalize_mean": tuple(normalize_mean),
            "normalize_std": tuple(normalize_std),
            "l2_storage_dtype": str(l2_dtype_str),
            "schema_version": schema_version,
        }
        cfg_str = "|".join(f"{k}={v}" for k, v in cfg_fields.items())
        return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return "00000000"

__all__ = ["compute_l2_cfg_hash"]
