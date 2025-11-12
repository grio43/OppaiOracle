#!/usr/bin/env python3
"""
Model Metadata Management
Handles embedding and extraction of vocabulary and preprocessing parameters
"""

import json
import gzip
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Manages model metadata including vocabulary and preprocessing params"""

    # Class-level cache for vocabulary data to avoid repeated disk I/O
    # Key: (vocab_path, mtime_ns) -> Value: (vocab_data, compressed_data)
    _vocab_cache: Dict[Tuple[str, int], Tuple[Dict, Dict[str, str]]] = {}

    @staticmethod
    def embed_vocabulary(checkpoint: Dict, vocab_path: Path) -> Dict:
        """Embed vocabulary into checkpoint

        Args:
            checkpoint: Checkpoint dictionary to modify
            vocab_path: Path to vocabulary JSON file

        Returns:
            Modified checkpoint with embedded vocabulary

        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            ValueError: If vocabulary is invalid or contains placeholders
            json.JSONDecodeError: If vocabulary file is not valid JSON
        """
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        try:
            # Check cache first to avoid repeated disk I/O (50-200ms saved per checkpoint)
            vocab_stat = vocab_path.stat()
            cache_key = (str(vocab_path.resolve()), vocab_stat.st_mtime_ns)

            if cache_key in ModelMetadata._vocab_cache:
                # Cache hit: use cached compressed data
                vocab_data, compressed_metadata = ModelMetadata._vocab_cache[cache_key]
                checkpoint.update(compressed_metadata)
                logger.debug(f"Using cached vocabulary ({len(vocab_data['tag_to_index'])} tags)")
                return checkpoint

            # Cache miss: load and validate vocabulary
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Verify vocabulary structure
            if 'tag_to_index' not in vocab_data or 'index_to_tag' not in vocab_data:
                raise ValueError(
                    f"Invalid vocabulary structure at {vocab_path}: "
                    f"missing required keys 'tag_to_index' or 'index_to_tag'"
                )

            # Check for placeholder tags
            placeholder_count = sum(
                1 for tag in vocab_data['tag_to_index'].keys()
                if tag.startswith("tag_") and tag[4:].isdigit()
            )
            if placeholder_count > 0:
                examples = [t for t in list(vocab_data['tag_to_index'].keys())[:10]
                           if t.startswith('tag_')]
                raise ValueError(
                    f"Vocabulary contains {placeholder_count} placeholder tags! "
                    f"Will not embed corrupted vocabulary. Examples: {examples}"
                )

            # Compress vocabulary (expensive operation, worth caching)
            vocab_json = json.dumps(vocab_data, ensure_ascii=False)
            vocab_bytes = vocab_json.encode('utf-8')
            vocab_compressed = gzip.compress(vocab_bytes)
            vocab_b64 = base64.b64encode(vocab_compressed).decode('utf-8')
            vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()

            # Create metadata dict for caching
            compressed_metadata = {
                'vocab_b64_gzip': vocab_b64,
                'vocab_format_version': '1',
                'vocab_sha256': vocab_sha256,
            }

            # Cache the result (limit cache size to prevent memory bloat)
            # Keep only the most recent 3 vocabularies (sufficient for multi-GPU training)
            if len(ModelMetadata._vocab_cache) >= 3:
                # Remove oldest entry
                oldest_key = next(iter(ModelMetadata._vocab_cache))
                del ModelMetadata._vocab_cache[oldest_key]

            ModelMetadata._vocab_cache[cache_key] = (vocab_data, compressed_metadata)

            # Embed in checkpoint
            checkpoint.update(compressed_metadata)

            logger.info(f"Embedded vocabulary with {len(vocab_data['tag_to_index'])} tags (cached for reuse)")
            return checkpoint

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in vocabulary file {vocab_path}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to embed vocabulary from {vocab_path}: {e}")
            raise

    @staticmethod
    def extract_vocabulary(checkpoint: Dict) -> Optional[Dict]:
        """Extract vocabulary from checkpoint

        Args:
            checkpoint: Checkpoint dictionary

        Returns:
            Vocabulary data or None if not found
        """
        if 'vocab_b64_gzip' not in checkpoint:
            return None

        try:
            vocab_b64 = checkpoint['vocab_b64_gzip']
            vocab_compressed = base64.b64decode(vocab_b64)
            vocab_bytes = gzip.decompress(vocab_compressed)

            # Verify integrity
            if 'vocab_sha256' in checkpoint:
                expected_sha = checkpoint['vocab_sha256']
                actual_sha = hashlib.sha256(vocab_bytes).hexdigest()
                if expected_sha != actual_sha:
                    logger.warning(f"Vocabulary SHA mismatch!")
                    return None

            vocab_json = vocab_bytes.decode('utf-8')
            vocab_data = json.loads(vocab_json)

            # Verify extracted vocabulary is valid
            placeholder_count = sum(
                1 for tag in vocab_data.get('tag_to_index', {}).keys()
                if tag.startswith("tag_") and tag[4:].isdigit()
            )
            if placeholder_count > 0:
                logger.error(f"Extracted vocabulary contains {placeholder_count} placeholder tags!")
                return None

            return vocab_data

        except Exception as e:
            logger.error(f"Failed to extract vocabulary: {e}")
            return None

    @staticmethod
    def embed_preprocessing_params(
        checkpoint: Dict,
        normalize_mean: Tuple[float, float, float],
        normalize_std: Tuple[float, float, float],
        image_size: int,
        patch_size: int
    ) -> Dict:
        """Embed preprocessing parameters into checkpoint"""

        preprocessing_params = {
            'normalize_mean': list(normalize_mean),
            'normalize_std': list(normalize_std),
            'image_size': image_size,
            'patch_size': patch_size,
        }

        checkpoint['preprocessing_params'] = preprocessing_params
        logger.info(f"Embedded preprocessing params: {preprocessing_params}")

        return checkpoint

    @staticmethod
    def extract_preprocessing_params(checkpoint: Dict) -> Optional[Dict]:
        """Extract preprocessing parameters from checkpoint"""

        if 'preprocessing_params' in checkpoint:
            return checkpoint['preprocessing_params']

        # Try legacy format
        if 'normalization_params' in checkpoint:
            legacy = checkpoint['normalization_params']
            return {
                'normalize_mean': legacy.get('mean', [0.5, 0.5, 0.5]),
                'normalize_std': legacy.get('std', [0.5, 0.5, 0.5]),
                'image_size': checkpoint.get('image_size', 640),
                'patch_size': checkpoint.get('patch_size', 16),
            }

        return None
