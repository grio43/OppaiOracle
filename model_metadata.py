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

    @staticmethod
    def embed_vocabulary(checkpoint: Dict, vocab_path: Path) -> Dict:
        """Embed vocabulary into checkpoint

        Args:
            checkpoint: Checkpoint dictionary to modify
            vocab_path: Path to vocabulary JSON file

        Returns:
            Modified checkpoint with embedded vocabulary
        """
        if not vocab_path.exists():
            logger.warning(f"Vocabulary file not found: {vocab_path}")
            return checkpoint

        try:
            # Verify the vocabulary file itself is valid before embedding
            with open(vocab_path, 'r', encoding='utf-8') as f:
                test_load = json.load(f)
                if not test_load.get('tag_to_index') or not test_load.get('index_to_tag'):
                    logger.error(f"Invalid vocabulary file structure at {vocab_path}")
                    return checkpoint

            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Verify vocabulary is valid
            if 'tag_to_index' not in vocab_data or 'index_to_tag' not in vocab_data:
                raise ValueError("Invalid vocabulary format")

            # Check for placeholder tags
            placeholder_count = sum(
                1 for tag in vocab_data['tag_to_index'].keys()
                if tag.startswith("tag_") and tag[4:].isdigit()
            )
            if placeholder_count > 0:
                logger.error(f"Vocabulary contains {placeholder_count} placeholder tags - not embedding!")
                logger.error(
                    f"Examples: {[t for t in list(vocab_data['tag_to_index'].keys())[:10] if t.startswith('tag_')]}"
                )
                raise ValueError(
                    f"Vocabulary contains {placeholder_count} placeholder tags! Will not embed corrupted vocabulary."
                )
                return checkpoint

            # Compress vocabulary
            vocab_json = json.dumps(vocab_data, ensure_ascii=False)
            vocab_bytes = vocab_json.encode('utf-8')
            vocab_compressed = gzip.compress(vocab_bytes)
            vocab_b64 = base64.b64encode(vocab_compressed).decode('utf-8')
            vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()

            # Embed in checkpoint
            checkpoint['vocab_b64_gzip'] = vocab_b64
            checkpoint['vocab_format_version'] = '1'
            checkpoint['vocab_sha256'] = vocab_sha256

            logger.info(f"Embedded vocabulary with {len(vocab_data['tag_to_index'])} tags")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to embed vocabulary: {e}")
            return checkpoint

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

