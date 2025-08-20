#!/usr/bin/env python3
"""
Shared vocabulary module for anime image tagger.

This module provides a unified TagVocabulary class that handles tag vocabulary
management for both training and inference, eliminating code duplication.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ignore tag handling
#
# Tags listed in a plain text file called ``Tags_ignore.txt`` located in the
# repository root will be automatically excluded from both the vocabulary and
# the training labels.  This allows callers to specify words that should be
# entirely ignored by the system – no learning signal is generated for them
# and they are omitted from scoring.  The list is loaded once when the
# ``TagVocabulary`` is constructed and stored in the module‑level
# ``IGNORE_TAG_INDICES`` so that other modules (e.g. metrics) can access
# the indices to be removed from predictions and targets.
#
# Each line in ``Tags_ignore.txt`` should contain a single tag.  Blank lines
# and lines beginning with ``#`` are ignored.

# Process-wide immutable default for ignored tag indices. DataLoader workers are separate
# processes; thread-local state does not propagate, leading to silent divergence. Keep
# a simple module-level tuple and pass copies into datasets.
from typing import Iterable, Tuple
from functools import lru_cache

IGNORE_TAG_INDICES: Tuple[int, ...] = tuple()

def get_ignore_tag_indices() -> Tuple[int, ...]:
    """Get the tuple of ignored tag indices."""
    return IGNORE_TAG_INDICES

def set_ignore_tag_indices(indices: Iterable[int]) -> None:
    """Set the ignored tag indices once at startup; avoids mutations during training."""
    global IGNORE_TAG_INDICES
    IGNORE_TAG_INDICES = tuple(sorted(set(int(i) for i in indices)))


def _load_ignore_tags(ignore_file: Optional[Path] = None) -> Set[str]:
    """Load ignore tags from a plain text file.

    If ``ignore_file`` is not provided, this attempts to locate a
    ``Tags_ignore.txt`` file in the same directory as this module.  Each
    non‑empty, non‑comment line of the file is treated as a tag to ignore.

    Returns:
        A set of tag strings that should be ignored.
    """
    # Determine the default ignore file location relative to this script.
    if ignore_file is None:
        # ``__file__`` points to this module; the ignore list resides in
        # the repository root alongside this file.  Using ``parent`` twice
        # ascends from ``vocabulary.py`` to the repository root.
        module_path = Path(__file__).resolve()
        ignore_file = module_path.parent / 'Tags_ignore.txt'
    ignored: Set[str] = set()
    try:
        if ignore_file and ignore_file.exists():
            with open(ignore_file, 'r', encoding='utf-8') as f:
                for line in f:
                    tag = line.strip()
                    # Skip blank lines or comments
                    if not tag or tag.startswith('#'):
                        continue
                    ignored.add(tag)
    except Exception as e:
        logger.warning(f"Failed to load ignore tags from {ignore_file}: {e}")
    return ignored


class TagVocabulary:
    """Unified tag vocabulary manager with support for both training and inference.
    
    This class combines functionality from both HDF5_loader.py and Inference_Engine.py
    to provide a single, comprehensive vocabulary implementation.
    """
    
    def __init__(self, vocab_path: Optional[Path] = None, min_frequency: int = 1) -> None:
        """Initialize vocabulary, optionally loading from file.
        
        Args:
            vocab_path: Optional path to vocabulary file to load
            min_frequency: Minimum frequency for tags to be included when building vocabulary
        """
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_frequencies: Dict[str, int] = {}
        self.min_frequency = min_frequency

        # Load the ignore list once.  ``ignored_tags`` holds the set of tag
        # strings that should be ignored entirely.  ``ignored_tag_indices`` is
        # derived later once the vocabulary is built or loaded.  These
        # attributes are instance specific so that unit tests or multiple
        # vocabularies can operate independently; however, the indices are also
        # broadcast via the module‑level ``IGNORE_TAG_INDICES`` so that
        # downstream code may mask out the ignored tags without a direct
        # reference to this instance.
        self.ignored_tags: Set[str] = _load_ignore_tags()
        self.ignored_tag_indices: List[int] = []
        
        # Special tokens+        # Use distinct strings for padding and unknown tokens.
        # The pad token (index 0) is reserved for masking and is not a valid output.
        # The unk token (index 1) represents unknown or rare tags.
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
        self.unk_index: int = 1  # Will be updated when vocabulary is built/loaded
        
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
        """Encode a list of tag strings into a multi-hot tensor.
        
        Unknown tags are mapped to the <UNK> index; the resulting tensor
        has shape (vocab_size,) and dtype float32.
        
        Args:
            tags: Iterable of tag strings
            
        Returns:
            Multi-hot tensor of shape (vocab_size,)
        """
        vector = torch.zeros(len(self.tag_to_index), dtype=torch.float32)
        # Skip any tags that are marked as ignored.  Ignored tags are not
        # encoded at all; they do not contribute to the label vector and
        # therefore produce no training signal.  Unknown tags are still
        # mapped to the <UNK> index as before.
        for tag in tags:
            if tag in self.ignored_tags:
                continue
            idx = self.tag_to_index.get(tag, self.tag_to_index.get(self.unk_token, self.unk_index))
            vector[idx] = 1.0
        return vector
    
    def get_tag_index(self, tag: str) -> int:
        """Get index for a tag (for inference compatibility).
        
        Args:
            tag: Tag string
            
        Returns:
            Index of the tag, or unk_index if not found
        """
        return self.tag_to_index.get(tag, self.unk_index)
    
    def get_tag_from_index(self, index: int) -> str:
        """Get tag from index (for inference compatibility).
        
        Args:
            index: Tag index
            
        Returns:
            Tag string, or <UNK> if index not found
        """
        return self.index_to_tag.get(index, self.unk_token)
    
    def build_from_annotations(self, json_files: List[Path], top_k: int) -> None:
        """Build a vocabulary from a collection of JSON annotation files.
        
        Args:
            json_files: List of annotation files to parse
            top_k: Maximum number of tags to keep (sorted by frequency)
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

                    # Accept both space‑delimited strings and lists
                    tags_list: List[str]
                    if isinstance(tags_field, str):
                        tags_list = tags_field.split()
                    elif isinstance(tags_field, list):
                        tags_list = tags_field
                    else:
                        continue

                    for tag in tags_list:
                        # Skip ignored tags entirely when building the vocabulary
                        if tag in self.ignored_tags:
                            continue
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

        # Update ignored tag indices and propagate to module‑level constant
        self.ignored_tag_indices = [self.tag_to_index.get(tag) for tag in self.ignored_tags if tag in self.tag_to_index]
        # Filter out any None values (in case an ignored tag did not make it into the vocab)
        self.ignored_tag_indices = [i for i in self.ignored_tag_indices if i is not None]
        # Use thread-local storage instead of global variable
        set_ignore_tag_indices(self.ignored_tag_indices)
        
        # Update tags list for compatibility
        self.tags = sorted_tags
        
        logger.info(f"Vocabulary built with {len(self.tag_to_index)} tags (incl. special tokens)")
    
    def save_vocabulary(self, vocab_path: Path) -> None:
        """Save the vocabulary to a JSON file.
        
        The file contains tag_to_index, index_to_tag, and tag_frequencies.
        
        Args:
            vocab_path: Path to save vocabulary file
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
        """Save vocabulary (alias for save_vocabulary for compatibility).
        
        Args:
            filepath: Path to save vocabulary file
        """
        self.save_vocabulary(filepath)
    
    def load_vocabulary(self, vocab_path: Path) -> None:
        """Load vocabulary from a JSON file.
        
        Args:
            vocab_path: Path to vocabulary JSON file
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
        
        # Update tags list for compatibility
        self.tags = [tag for tag in self.tag_to_index.keys() 
                     if tag not in (self.pad_token, self.unk_token)]
        
        # Compute ignored tag indices based on the loaded vocabulary.
        self.ignored_tag_indices = [self.tag_to_index.get(tag) for tag in self.ignored_tags if tag in self.tag_to_index]
        # Filter out missing tags (None values) in case ignored tags are absent
        self.ignored_tag_indices = [i for i in self.ignored_tag_indices if i is not None]
        # Use thread-local storage instead of a global variable so that downstream modules can mask these indices
        set_ignore_tag_indices(self.ignored_tag_indices)

        logger.info(f"Loaded vocabulary with {len(self.tag_to_index)} tags from {vocab_path}")
    
    @classmethod
    def from_file(cls, filepath: Path) -> 'TagVocabulary':
        """Load vocabulary from a simple text file (one tag per line).
        
        This method is for backward compatibility with inference code.
        
        Args:
            filepath: Path to text file with tags
            
        Returns:
            TagVocabulary instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = [line.strip() for line in f if line.strip()]
        
        vocab = cls()
        
        # Build vocabulary from tags list
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
    
    First tries to load from vocabulary.json, then tags.txt, 
    otherwise creates a dummy vocabulary.
    
    Args:
        vocab_dir: Directory containing vocabulary files
        
    Returns:
        TagVocabulary instance
    """
    # Try JSON format first
    vocab_json = vocab_dir / "vocabulary.json"
    if vocab_json.exists():
        vocab = TagVocabulary()
        vocab.load_vocabulary(vocab_json)
        return vocab
    
    # Try text format
    vocab_file = vocab_dir / "tags.txt"
    if vocab_file.exists():
        return TagVocabulary.from_file(vocab_file)
    
    # Create dummy vocabulary for demo
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