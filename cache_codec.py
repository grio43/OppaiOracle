from __future__ import annotations
import os
import hmac
import hashlib
from typing import Optional

import torch
from safetensors.torch import save, load

# Optional HMAC key for integrity checks
_HMAC_KEY = os.environ.get("CACHE_CODEC_HMAC_KEY")
_DIGEST_SIZE = hashlib.sha256().digest_size


def _resolve_key(key: Optional[bytes | str]) -> Optional[bytes]:
    if key is None:
        key = _HMAC_KEY
    if key is None:
        return None
    if isinstance(key, str):
        return key.encode("utf-8")
    return key


def encode_tensor(t: torch.Tensor, key: Optional[bytes | str] = None) -> bytes:
    """
    Serialize a tensor to bytes using safetensors with optional HMAC.

    The tensor is detached and moved to CPU before encoding to avoid
    capturing device-specific state.  When a key is provided, an HMAC
    digest of the payload is prepended.
    """
    t_cpu = t.detach().cpu().contiguous()
    data = save({"t": t_cpu})
    k = _resolve_key(key)
    if k:
        digest = hmac.new(k, data, "sha256").digest()
        data = digest + data
    return data


def decode_tensor(b: bytes, key: Optional[bytes | str] = None) -> torch.Tensor:
    """
    Deserialize bytes back into a tensor, verifying HMAC if present.
    With a key, the payload must be at least one digest long; the digest
    is recomputed and compared using compare_digest() to mitigate timing
    attacks.  Without a key, it first tries to load the payload directly,
    then falls back to skipping a digest prefix for legacy entries.
    """
    k = _resolve_key(key)
    data = b
    if k:
        if len(data) < _DIGEST_SIZE:
            raise ValueError("Payload too short for HMAC digest")
        digest, payload = data[:_DIGEST_SIZE], data[_DIGEST_SIZE:]
        expected = hmac.new(k, payload, "sha256").digest()
        if not hmac.compare_digest(digest, expected):
            raise ValueError("HMAC verification failed")
        data = payload
        return load(data)["t"]
    # Non-HMAC: try direct load, then legacy fallback
    try:
        return load(data)["t"]
    except (RuntimeError, KeyError) as e:
        # safetensors.load raises RuntimeError for invalid format
        # KeyError if "t" key missing
        # Try legacy fallback if data is long enough
        if len(data) > _DIGEST_SIZE:
            try:
                return load(data[_DIGEST_SIZE:])["t"]
            except (RuntimeError, KeyError):
                pass  # Fallback failed
        # Re-raise original error
        raise ValueError(f"Failed to decode cache entry: {e}") from e


__all__ = ["encode_tensor", "decode_tensor"]
