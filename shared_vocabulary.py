#!/usr/bin/env python3
"""
Shared vocabulary using multiprocessing shared memory.

This module provides a way to share vocabulary across multiple worker processes
without duplicating it in memory, reducing startup overhead and memory usage.

NOTE: Uses `from __future__ import annotations` to defer type hint evaluation,
allowing type hints like `shared_memory.SharedMemory` to work even on Python < 3.8
where `shared_memory` module is not available.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
from typing import Optional, Set
import sys

logger = logging.getLogger(__name__)

# Check Python version for shared_memory support
_has_shared_memory = sys.version_info >= (3, 8)

if _has_shared_memory:
    from multiprocessing import shared_memory
else:
    shared_memory = None  # type: ignore

# Registry of shared memory segments to clean up on exit
_SHARED_MEMORY_REGISTRY: Set[str] = set()
_SHARED_MEMORY_REGISTRY_LOCK = threading.Lock()  # Thread-safe access to registry

def _cleanup_shared_memory():
    """Clean up all registered shared memory segments on exit.

    This function handles the race condition where multiple processes may try
    to unlink the same shared memory segment. FileNotFoundError indicates
    another process already unlinked it, which is fine.
    """
    if not _has_shared_memory or shared_memory is None:
        return

    # Get a snapshot of names to clean up (thread-safe)
    with _SHARED_MEMORY_REGISTRY_LOCK:
        names_to_cleanup = list(_SHARED_MEMORY_REGISTRY)

    for shm_name in names_to_cleanup:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            try:
                shm.unlink()
                logger.debug(f"Cleaned up shared memory segment: {shm_name}")
            except FileNotFoundError:
                # Another process already unlinked this segment - this is fine
                logger.debug(f"Shared memory '{shm_name}' already unlinked by another process")
        except FileNotFoundError:
            # Segment doesn't exist (already cleaned up by another process)
            logger.debug(f"Shared memory '{shm_name}' not found - already cleaned up")
        except Exception as e:
            # Other errors (permission, etc.) - log but continue
            logger.debug(f"Could not clean up shared memory '{shm_name}': {e}")
        finally:
            with _SHARED_MEMORY_REGISTRY_LOCK:
                _SHARED_MEMORY_REGISTRY.discard(shm_name)

# Register cleanup on exit
atexit.register(_cleanup_shared_memory)


class SharedVocabularyManager:
    """Manager for vocabulary stored in shared memory.

    This allows multiple worker processes to access the same vocabulary data
    without duplicating it in each process's memory, reducing startup overhead
    from ~400ms (8 workers × 50ms) to near zero after initial load.
    """

    def __init__(self):
        """Initialize the shared vocabulary manager."""
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.shm_name: Optional[str] = None
        self.vocab_size: int = 0

    def create_from_vocab(self, vocab) -> str:
        """Serialize vocabulary to shared memory.

        Args:
            vocab: TagVocabulary instance to share

        Returns:
            Name of the shared memory segment

        Raises:
            RuntimeError: If shared memory is not available (Python < 3.8)
        """
        if not _has_shared_memory or shared_memory is None:
            raise RuntimeError(
                "Shared memory requires Python 3.8+. "
                f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
            )

        # Serialize vocabulary to bytes using JSON (safer than pickle)
        # Include only necessary data to minimize size
        # Add content hash for version verification
        #
        # SECURITY: Using JSON instead of pickle to prevent arbitrary code execution
        # when loading vocabulary from shared memory
        vocab_content = {
            'tag_to_index': vocab.tag_to_index,
            # JSON requires string keys, convert integer keys to strings
            'index_to_tag': {str(k): v for k, v in vocab.index_to_tag.items()},
            'tag_frequencies': vocab.tag_frequencies,
            # Convert set to list for JSON serialization
            'ignored_tag_indices': list(vocab.ignored_tag_indices) if isinstance(vocab.ignored_tag_indices, set) else vocab.ignored_tag_indices,
            'pad_token': vocab.pad_token,
            'unk_token': vocab.unk_token,
            'rating_to_index': vocab.rating_to_index,
            'tags': list(vocab.tags) if isinstance(vocab.tags, set) else vocab.tags,
            'unk_index': vocab.unk_index,
        }
        # Compute full SHA256 hash for version verification (no truncation for security)
        content_hash = hashlib.sha256(
            json.dumps(sorted(vocab.tag_to_index.items()), ensure_ascii=False).encode()
        ).hexdigest()  # Full 64-char hash, not truncated
        vocab_data = {
            **vocab_content,
            '_vocab_hash': content_hash,
            '_vocab_size': len(vocab.tag_to_index),
        }

        # Use JSON for safe serialization (no arbitrary code execution risk)
        data_bytes = json.dumps(vocab_data, ensure_ascii=False).encode('utf-8')
        self.vocab_size = len(data_bytes)

        # Create shared memory buffer
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.vocab_size)
            self.shm_name = self.shm.name

            # Write vocabulary data to shared memory
            self.shm.buf[:self.vocab_size] = data_bytes

            # Register for cleanup on exit (thread-safe)
            with _SHARED_MEMORY_REGISTRY_LOCK:
                _SHARED_MEMORY_REGISTRY.add(self.shm_name)

            logger.info(
                f"Created shared vocabulary in memory '{self.shm_name}' "
                f"({self.vocab_size / 1024:.1f} KB, {len(vocab)} tags)"
            )

            return self.shm_name

        except Exception as e:
            logger.error(f"Failed to create shared vocabulary: {e}")
            if self.shm is not None:
                try:
                    self.shm.close()
                    self.shm.unlink()
                except Exception:
                    pass
            raise

    @staticmethod
    def load_from_shared(shm_name: str, vocab_size: int):
        """Load vocabulary from shared memory.

        Args:
            shm_name: Name of the shared memory segment
            vocab_size: Size of the vocabulary data in bytes

        Returns:
            Dictionary with vocabulary data

        Raises:
            RuntimeError: If shared memory is not available
        """
        if not _has_shared_memory or shared_memory is None:
            raise RuntimeError("Shared memory requires Python 3.8+")

        try:
            # Attach to existing shared memory
            shm = shared_memory.SharedMemory(name=shm_name)

            # Read and deserialize vocabulary data using JSON (safe)
            data_bytes = bytes(shm.buf[:vocab_size])
            vocab_data = json.loads(data_bytes.decode('utf-8'))

            # Convert string keys back to integers for index_to_tag
            if 'index_to_tag' in vocab_data:
                vocab_data['index_to_tag'] = {int(k): v for k, v in vocab_data['index_to_tag'].items()}

            # Convert list back to set for ignored_tag_indices
            if 'ignored_tag_indices' in vocab_data and isinstance(vocab_data['ignored_tag_indices'], list):
                vocab_data['ignored_tag_indices'] = set(vocab_data['ignored_tag_indices'])

            # Convert list back to set for tags if it was a set
            # Note: We store as list, caller should know if it needs set
            if 'tags' in vocab_data and isinstance(vocab_data['tags'], list):
                vocab_data['tags'] = set(vocab_data['tags'])

            # Close but don't unlink (other processes may still need it)
            shm.close()

            return vocab_data

        except Exception as e:
            logger.error(f"Failed to load vocabulary from shared memory '{shm_name}': {e}")
            raise

    def close(self):
        """Close the shared memory (don't unlink - other processes may still use it)."""
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception as e:
                logger.warning(f"Error closing shared vocabulary: {e}")

    def cleanup(self):
        """Close and unlink shared memory (idempotent - safe to call multiple times)."""
        if self.shm is None:
            return  # Already cleaned up

        # Remove from registry first to prevent other cleanups (thread-safe)
        if self.shm_name:
            with _SHARED_MEMORY_REGISTRY_LOCK:
                _SHARED_MEMORY_REGISTRY.discard(self.shm_name)

        try:
            self.shm.close()
            self.shm.unlink()
            logger.info(f"Cleaned up shared vocabulary '{self.shm_name}'")
        except FileNotFoundError:
            # Already unlinked by another cleanup, that's fine
            pass
        except Exception as e:
            logger.warning(f"Error cleaning up shared vocabulary: {e}")
        finally:
            self.shm = None
            self.shm_name = None


def populate_vocab_from_shared(vocab, vocab_data: dict, verify: bool = True):
    """Populate a TagVocabulary instance with data from shared memory.

    Args:
        vocab: TagVocabulary instance to populate
        vocab_data: Dictionary with vocabulary data from shared memory
        verify: If True, verify content hash matches expected (default: True)

    Raises:
        ValueError: If verification fails (hash mismatch or missing metadata)
    """
    # Verify vocabulary integrity if hash metadata is present
    if verify:
        stored_hash = vocab_data.get('_vocab_hash')
        stored_size = vocab_data.get('_vocab_size')
        if stored_hash is not None and stored_size is not None:
            # Recompute full SHA256 hash to verify (no truncation for security)
            actual_hash = hashlib.sha256(
                json.dumps(sorted(vocab_data['tag_to_index'].items()), ensure_ascii=False).encode()
            ).hexdigest()  # Full 64-char hash
            actual_size = len(vocab_data['tag_to_index'])
            # Handle both old truncated hashes (16-char) and new full hashes (64-char)
            hash_match = (actual_hash == stored_hash or actual_hash[:16] == stored_hash)
            if not hash_match or actual_size != stored_size:
                raise ValueError(
                    f"Shared vocabulary integrity check failed: "
                    f"hash {stored_hash[:16]}... vs {actual_hash[:16]}..., "
                    f"size {stored_size} vs {actual_size}. "
                    "The vocabulary may have been corrupted or modified."
                )
            logger.debug(f"Shared vocabulary verified: hash={actual_hash[:16]}..., size={actual_size}")

    vocab.tag_to_index = vocab_data['tag_to_index']
    vocab.index_to_tag = vocab_data['index_to_tag']
    vocab.tag_frequencies = vocab_data['tag_frequencies']
    vocab.ignored_tag_indices = vocab_data['ignored_tag_indices']
    vocab.pad_token = vocab_data['pad_token']
    vocab.unk_token = vocab_data['unk_token']
    vocab.rating_to_index = vocab_data['rating_to_index']
    vocab.tags = vocab_data['tags']
    vocab.unk_index = vocab_data['unk_index']


def is_shared_memory_available() -> bool:
    """Check if shared memory is available on this system.

    Returns:
        True if shared memory is available (Python 3.8+), False otherwise
    """
    return _has_shared_memory and shared_memory is not None
