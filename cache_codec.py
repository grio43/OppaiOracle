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
    """Serialize a tensor to bytes using safetensors with optional HMAC."""
    data = save({"t": t.contiguous()})
    k = _resolve_key(key)
    if k:
        digest = hmac.new(k, data, "sha256").digest()
        data = digest + data
    return data


def decode_tensor(b: bytes, key: Optional[bytes | str] = None) -> torch.Tensor:
    """Deserialize bytes back into a tensor, verifying HMAC if present."""
    k = _resolve_key(key)
    data = b
    if k:
        digest, payload = data[:_DIGEST_SIZE], data[_DIGEST_SIZE:]
        expected = hmac.new(k, payload, "sha256").digest()
        if not hmac.compare_digest(digest, expected):
            raise ValueError("HMAC verification failed")
        data = payload
        tensors = load(data)
        return tensors["t"]
    # No key provided: try plain payload first; if that fails, also try skipping a possible HMAC prefix
    try:
        tensors = load(data)
        return tensors["t"]
    except Exception:
        try:
            tensors = load(data[_DIGEST_SIZE:])
            return tensors["t"]
        except Exception:
            # propagate original error for clarity
            raise


__all__ = ["encode_tensor", "decode_tensor"]
