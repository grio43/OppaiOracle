from __future__ import annotations
import os
import hmac
import hashlib
import pickle
import io
from typing import Optional

import torch
from safetensors.torch import save, load

# Optional HMAC key for integrity checks
_HMAC_KEY = os.environ.get("CACHE_CODEC_HMAC_KEY")
_DIGEST_SIZE = hashlib.sha256().digest_size

# Threshold for using pickle vs safetensors (1KB)
# Below this size, pickle has less overhead than safetensors
_SMALL_TENSOR_THRESHOLD = 1024

# Format markers for adaptive codec
_FORMAT_PICKLE = b'P'
_FORMAT_SAFETENSORS = b'S'


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
    Serialize a tensor to bytes using adaptive codec with optional HMAC.

    Uses pickle for small tensors (< 1KB) and safetensors for larger ones.
    This reduces overhead for small tensors while maintaining safety for large ones.

    The tensor is detached and moved to CPU before encoding to avoid
    capturing device-specific state.  When a key is provided, an HMAC
    digest of the payload is prepended.

    Format:
    - [HMAC digest (32 bytes, if key provided)] + [format marker (1 byte)] + [payload]
    """
    t_cpu = t.detach().cpu().contiguous()
    size_bytes = t_cpu.element_size() * t_cpu.nelement()

    # Choose serialization method based on tensor size
    if size_bytes < _SMALL_TENSOR_THRESHOLD:
        # Use pickle for small tensors (less metadata overhead)
        payload = pickle.dumps(t_cpu, protocol=pickle.HIGHEST_PROTOCOL)
        data = _FORMAT_PICKLE + payload
    else:
        # Use safetensors for large tensors (better safety and performance)
        buffer = io.BytesIO()
        save({"t": t_cpu}, buffer)
        data = _FORMAT_SAFETENSORS + buffer.getvalue()

    # Add HMAC if key provided
    k = _resolve_key(key)
    if k:
        digest = hmac.new(k, data, "sha256").digest()
        data = digest + data
    return data


def decode_tensor(b: bytes, key: Optional[bytes | str] = None) -> torch.Tensor:
    """
    Deserialize bytes back into a tensor, verifying HMAC if present.

    Supports both pickle and safetensors formats based on format marker.
    Also maintains backward compatibility with old safetensors-only format.

    With a key, the payload must be at least one digest long; the digest
    is recomputed and compared using compare_digest() to mitigate timing
    attacks.  Without a key, it first tries to load the payload directly,
    then falls back to skipping a digest prefix for legacy entries.
    """
    k = _resolve_key(key)
    data = b

    # HMAC verification if key provided
    if k:
        if len(data) < _DIGEST_SIZE:
            raise ValueError("Payload too short for HMAC digest")
        digest, payload = data[:_DIGEST_SIZE], data[_DIGEST_SIZE:]
        expected = hmac.new(k, payload, "sha256").digest()
        if not hmac.compare_digest(digest, expected):
            raise ValueError("HMAC verification failed")
        data = payload

    # Check format marker and decode accordingly
    if len(data) > 0:
        format_marker = data[0:1]
        if format_marker == _FORMAT_PICKLE:
            # Pickle format
            return pickle.loads(data[1:])
        elif format_marker == _FORMAT_SAFETENSORS:
            # Safetensors format with marker
            buffer = io.BytesIO(data[1:])
            return load(buffer)["t"]

    # Legacy format fallback: try direct safetensors load (old format without marker)
    try:
        return load(data)["t"]
    except (RuntimeError, KeyError) as e:
        # safetensors.load raises RuntimeError for invalid format
        # KeyError if "t" key missing
        # Try legacy fallback with HMAC digest skip if data is long enough
        if len(data) > _DIGEST_SIZE:
            try:
                return load(data[_DIGEST_SIZE:])["t"]
            except (RuntimeError, KeyError):
                pass  # Fallback failed
        # Re-raise original error
        raise ValueError(f"Failed to decode cache entry: {e}") from e


__all__ = ["encode_tensor", "decode_tensor"]
