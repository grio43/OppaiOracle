#!/usr/bin/env python3
"""
Shared vocabulary module for anime image tagger.

This module provides a unified TagVocabulary class that handles tag
management for both training and inference.  The pad token at index 0
is reserved for masking purposes and **is not** a valid output class.
Downstream loss functions and metrics must ignore this index when
computing gradients or scores.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import torch

logger = logging.getLogger(__name__)


class TagVocabulary:
    """Unified tag vocabulary manager with support for both training and inference.

    This class combines functionality from both data loading and inference
    code to provide a single, comprehensive vocabulary implementation.
    The special ``pad_token`` occupies index 0 and is used solely for
    padding/masking; it should never be considered a valid label during
    training or evaluation.  The ``unk_token`` at index 1 is used to
    represent unknown or rare tags.
    """

    def __init__(self, vocab_path: Optional[Path] = None, min_frequency: int = 1) -> None:
        """Initialize vocabulary, optionally loading from file.

        Args:
            vocab_path: Optional path to vocabulary file to load.
            min_frequency: Minimum frequency for tags to be included when building vocabulary.
        """
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_frequencies: Dict[str, int] = {}
        self.min_frequency = min_frequency

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # Rating classes (fixed)
        self.rating_to_index: Dict[str, int] = {
            "general": 0,
            "sensitive": 1,
            "questionable": 2,
            "explicit": 3,
            "unknown": 4,
        }

        # Convenience attributes for compatibility
        self.tags: List[str] = []
        # Unknown index will be updated when vocabulary is built/loaded
        self.unk_index: int = 1

        # If a vocabulary file is supplied, attempt to load it
        if vocab_path is not None and vocab_path.exists():
            try:
                self.load_vocabulary(vocab_path)
            except Exception:
                logger.info(f"Could not load vocabulary from {vocab_path}, will build a new one")

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.tag_to_index)

    def encode_tags(self, tags: Iterable[str]) -> torch.Tensor:
        """Encode a list of tag strings into a multi‑hot tensor.

        Unknown tags are mapped to the ``<UNK>`` index; the resulting tensor
        has shape ``(vocab_size,)`` and dtype ``float32``.

        Args:
            tags: Iterable of tag strings.

        Returns:
            Multi‑hot tensor of shape ``(vocab_size,)``.
        """
        vector = torch.zeros(len(self.tag_to_index), dtype=torch.float32)
        for tag in tags:
            idx = self.tag_to_index.get(tag, self.tag_to_index.get(self.unk_token, self.unk_index))
            vector[idx] = 1.0
        return vector

    def get_tag_index(self, tag: str) -> int:
        """Get index for a tag (for inference compatibility).

        Args:
            tag: Tag string.

        Returns:
            Index of the tag, or ``unk_index`` if not found.
        """
        return self.tag_to_index.get(tag, self.unk_index)

    def get_tag_from_index(self, index: int) -> str:
        """Get tag from index (for inference compatibility).

        Args:
            index: Tag index.

        Returns:
            Tag string, or ``<UNK>`` if index not found.
        """
        return self.index_to_tag.get(index, self.unk_token)

    def build_from_annotations(self, json_files: List[Path], top_k: Optional[int]) -> None:
        """Build a vocabulary from a collection of JSON annotation files.

        Args:
            json_files: List of annotation files to parse.
            top_k: Maximum number of tags to keep (sorted by frequency).  If ``None``,
                all tags meeting ``min_frequency`` will be included.
        """
        logger.info(f"Building vocabulary from {len(json_files)} annotation files")
        tag_counts: Dict[str, int] = {}
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for entry in data:
                    tags_field = entry.get('tags')
                    if not tags_field:
                        continue
                    tags_list: List[str]
                    if isinstance(tags_field, str):
                        tags_list = tags_field.split()
                    elif isinstance(tags_field, list):
                        tags_list = tags_field
                    else:
                        continue
                    for tag in tags_list:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
        # Sort tags by frequency and cut to top_k
        sorted_tags = sorted(
            [t for t, c in tag_counts.items() if c >= self.min_frequency],
            key=lambda x: (-tag_counts[x], x)
        )
        if top_k is not None and top_k > 0:
            sorted_tags = sorted_tags[:top_k]
        # Assign indices. Reserve 0 for <PAD> and 1 for <UNK>
        self.tag_to_index = {self.pad_token: 0, self.unk_token: 1}
        self.index_to_tag = {0: self.pad_token, 1: self.unk_token}
        self.unk_index = 1
        for idx, tag in enumerate(sorted_tags, start=2):
            self.tag_to_index[tag] = idx
            self.index_to_tag[idx] = tag
            self.tag_frequencies[tag] = tag_counts[tag]
        self.tags = sorted_tags
        logger.info(f"Vocabulary built with {len(self.tag_to_index)} tags (including special tokens)")

    def save_vocabulary(self, vocab_path: Path) -> None:
        """Save the vocabulary to a JSON file.

        The file contains ``tag_to_index``, ``index_to_tag`` and ``tag_frequencies``.

        Args:
            vocab_path: Path to save vocabulary file.
        """
        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'tag_to_index': self.tag_to_index,
                'index_to_tag': self.index_to_tag,
                'tag_frequencies': self.tag_frequencies,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved vocabulary to {vocab_path}")

    def save(self, filepath: Path) -> None:
        """Alias for :meth:`save_vocabulary` for compatibility."""
        self.save_vocabulary(filepath)

    def load_vocabulary(self, vocab_path: Path) -> None:
        """Load vocabulary from a JSON file.

        Args:
            vocab_path: Path to vocabulary JSON file.
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tag_to_index = data['tag_to_index']
        self.index_to_tag = {int(k): v for k, v in data['index_to_tag'].items()}
        self.tag_frequencies = data.get('tag_frequencies', {})
        # Ensure special tokens are present
        for token in (self.pad_token, self.unk_token):
            if token not in self.tag_to_index:
                idx = len(self.tag_to_index)
                self.tag_to_index[token] = idx
                self.index_to_tag[idx] = token
        # Update unk_index
        self.unk_index = self.tag_to_index.get(self.unk_token, 1)
        # Update tags list for compatibility (exclude special tokens)
        self.tags = [tag for tag in self.tag_to_index.keys() if tag not in (self.pad_token, self.unk_token)]
        logger.info(f"Loaded vocabulary with {len(self.tag_to_index)} tags from {vocab_path}")

    @classmethod
    def from_file(cls, filepath: Path) -> 'TagVocabulary':
        """Load vocabulary from a simple text file (one tag per line).

        This method is for backward compatibility with inference code.

        Args:
            filepath: Path to text file with tags.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = [line.strip() for line in f if line.strip()]
        vocab = cls()
        vocab.tag_to_index = {vocab.pad_token: 0, vocab.unk_token: 1}
        vocab.index_to_tag = {0: vocab.pad_token, 1: vocab.unk_token}
        vocab.unk_index = 1
        for idx, tag in enumerate(tags, start=2):
            vocab.tag_to_index[tag] = idx
            vocab.index_to_tag[idx] = tag
        vocab.tags = tags
        return vocab

def load_vocabulary_for_training(vocab_dir: Path) -> TagVocabulary:
    """Load vocabulary from directory (for backward compatibility).

    First tries to load from ``vocabulary.json``, then ``tags.txt``,
    otherwise creates a dummy vocabulary.

    Args:
        vocab_dir: Directory containing vocabulary files.

    Returns:
        TagVocabulary instance.
    """
    vocab_json = vocab_dir / "vocabulary.json"
    if vocab_json.exists():
        vocab = TagVocabulary()
        vocab.load_vocabulary(vocab_json)
        return vocab
    vocab_file = vocab_dir / "tags.txt"
    if vocab_file.exists():
        return TagVocabulary.from_file(vocab_file)
    logger.warning(f"Vocabulary file not found in {vocab_dir}, using dummy vocabulary")
    dummy_tags = [f"tag_{i}" for i in range(1000)]
    vocab = TagVocabulary()
    vocab.tag_to_index = {vocab.pad_token: 0, vocab.unk_token: 1}
    vocab.index_to_tag = {0: vocab.pad_token, 1: vocab.unk_token}
    vocab.unk_index = 1
    for idx, tag in enumerate(dummy_tags, start=2):
        vocab.tag_to_index[tag] = idx
        vocab.index_to_tag[idx] = tag
    vocab.tags = dummy_tags
    return vocab