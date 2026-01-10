#!/usr/bin/env python3
"""
Shared vocabulary module for anime image tagger.

This module provides a unified TagVocabulary class that handles tag vocabulary
management for both training and inference, eliminating code duplication.
"""

import atexit
import json
import logging
import itertools
import threading

# Try to use orjson for faster JSON parsing (3-5x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# Define JSON decode error types for exception handling (orjson has its own error type)
_JSON_ERRORS = (json.JSONDecodeError, orjson.JSONDecodeError) if HAS_ORJSON else (json.JSONDecodeError,)

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union
import torch
import yaml

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
VOCAB_PATH = PROJECT_ROOT / "vocabulary.json"

# Global vocabulary cache to avoid reloading same vocabulary
_VOCAB_CACHE: Dict[str, 'TagVocabulary'] = {}
_VOCAB_CACHE_LOCK = threading.Lock()  # Thread-safe access to vocabulary cache

def _clear_vocab_cache():
    """Clear the vocabulary cache (called on exit)."""
    global _VOCAB_CACHE
    with _VOCAB_CACHE_LOCK:
        _VOCAB_CACHE.clear()

# Register cleanup on exit
atexit.register(_clear_vocab_cache)

def _load_ignore_tags(ignore_file: Optional[Path] = None) -> Set[str]:
    """Load ignore tags from a plain text file.

    Each non-empty, non-comment line is treated as a tag to ignore.
    Falls back to Tags_ignore.txt in this module's directory if no path given.
    """
    if ignore_file is None:
        module_path = Path(__file__).resolve()
        ignore_file = module_path.parent / 'Tags_ignore.txt'

    ignored: Set[str] = set()

    # Validate file exists (CR-013)
    if not ignore_file.exists():
        logger.debug(f"Ignored tags file not found: {ignore_file}")
        return ignored

    # Prevent DoS: check file size (CR-013)
    try:
        file_size = ignore_file.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            logger.error(f"Ignored tags file too large: {file_size} bytes (max 10MB)")
            return ignored
    except Exception as e:
        logger.error(f"Error checking file size for {ignore_file}: {e}")
        return ignored

    try:
        with open(ignore_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                tag = line.strip()
                # Skip blank lines or comments
                if not tag or tag.startswith('#'):
                    continue

                # Validate tag format (CR-013)
                if len(tag) > 200:  # Reasonable tag length
                    logger.warning(f"Suspicious tag length at line {line_num}: {len(tag)} chars")
                    continue

                if not tag.isprintable():
                    logger.warning(f"Non-printable characters in tag at line {line_num}: {tag!r}")
                    continue

                ignored.add(tag)

        logger.info(f"Loaded {len(ignored)} ignored tags from {ignore_file}")
    except Exception as e:
        logger.error(f"Failed to load ignore tags from {ignore_file}: {e}")

    return ignored


def _iter_chunks(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _count_tags_in_files(file_paths: List[str], ignored_tags: Set[str]) -> Dict[str, int]:
    tag_counts: Counter = Counter()
    for json_file in file_paths:
        try:
            # Use orjson if available (3-5x faster for large-scale vocabulary building)
            if HAS_ORJSON:
                data = orjson.loads(Path(json_file).read_bytes())
            else:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            entries = data if isinstance(data, list) else [data]

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                tags_field = entry.get('tags')
                if not tags_field:
                    continue

                if isinstance(tags_field, str):
                    tags_list = [tag.strip() for tag in tags_field.split(',') if tag.strip()]
                elif isinstance(tags_field, list):
                    tags_list = [str(tag).strip() for tag in tags_field if str(tag).strip()]
                else:
                    continue

                for tag in tags_list:
                    if tag in ignored_tags:
                        continue
                    tag_counts[tag] += 1

        except (KeyError, *_JSON_ERRORS) as e:
            logger.warning(f"Failed to parse {json_file}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error parsing {json_file}: {e}")

    return tag_counts


class TagVocabulary:
    """Unified tag vocabulary manager for training and inference."""
    
    def __init__(self,
                 vocab_path: Optional[Path] = None,
                 min_frequency: int = 1,
                 ignore_file: Optional[Path] = None,
                 pad_token: str = "<PAD>",
                 unk_token: str = "<UNK>",
                 tag_vector_dtype: str = "bfloat16") -> None:
        """Initialize vocabulary, optionally loading from file.

        Args:
            vocab_path: Optional path to vocabulary file to load
            min_frequency: Minimum frequency for tags to be included when building vocabulary
            ignore_file: Path to a file containing tags to ignore
            pad_token: The token to use for padding
            unk_token: The token to use for unknown tags
            tag_vector_dtype: Dtype for tag vectors ('float16', 'bfloat16', 'float32')
        """
        self.tag_to_index: Dict[str, int] = {}
        self.index_to_tag: Dict[int, str] = {}
        self.tag_frequencies: Dict[str, int] = {}
        self.min_frequency = min_frequency
        self.ignored_tags: Set[str] = _load_ignore_tags(ignore_file)
        self.ignored_tag_indices: List[int] = []
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Tag vector dtype configuration
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._tag_vector_dtype = dtype_map.get(str(tag_vector_dtype).lower(), torch.bfloat16)

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
            except Exception as e:
                # Critical: vocabulary load failure should not be silent
                # An empty vocabulary will cause training to produce meaningless results
                logger.error(f"Failed to load vocabulary from {vocab_path}: {e}")
                raise ValueError(f"Vocabulary load failed for {vocab_path}: {e}") from e
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.tag_to_index)
    
    def get_ignored_indices(self) -> List[int]:
        """Get the list of ignored tag indices.
        
        This should be passed to dataset constructors when using multiprocessing.
        """
        return self.ignored_tag_indices.copy()
    
    def encode_tags(self, tags: Iterable[str]) -> torch.Tensor:
        """Encode tag strings into a multi-hot tensor of shape (vocab_size,)."""
        vector = torch.zeros(len(self.tag_to_index), dtype=self._tag_vector_dtype)
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
        """Get tag from index. Returns <UNK> if not found."""
        tag = self.index_to_tag.get(index, self.unk_token)
        # Fail fast if we encounter a placeholder tag
        if tag != self.unk_token and tag.startswith("tag_") and len(tag) > 4 and tag[4:].isdigit():
            raise ValueError(
                f"CRITICAL: Placeholder tag '{tag}' detected at index {index}. "
                f"The vocabulary is corrupted. Please regenerate from dataset annotations."
            )
        return tag
    
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
                # Use orjson if available (3-5x faster for large-scale vocabulary building)
                if HAS_ORJSON:
                    data = orjson.loads(Path(json_file).read_bytes())
                else:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                # Handle both single entry (dict) and multiple entries (list)
                entries = data if isinstance(data, list) else [data]

                for entry in entries:
                    # Extract tags field from entry
                    if not isinstance(entry, dict):
                        logger.debug(f"Skipping non-dict entry in {json_file}")
                        continue

                    tags_field = entry.get('tags')
                    if not tags_field:
                        continue

                    # Accept both comma‑delimited strings and lists
                    tags_list: List[str]
                    if isinstance(tags_field, str):
                        tags_list = [tag.strip() for tag in tags_field.split(',') if tag.strip()]
                    elif isinstance(tags_field, list):
                        tags_list = [str(tag).strip() for tag in tags_field if str(tag).strip()]
                    else:
                        logger.debug(f"Skipping entry with non-string/list tags in {json_file}")
                        continue

                    for tag in tags_list:
                        # Skip ignored tags entirely when building the vocabulary
                        if tag in self.ignored_tags:
                            continue
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            except (KeyError, *_JSON_ERRORS) as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error parsing {json_file}: {e}")

        self.build_from_tag_counts(tag_counts, top_k)

    def build_from_tag_counts(self, tag_counts: Dict[str, int], top_k: Optional[int]) -> None:
        """Build a vocabulary from precomputed tag counts.

        Args:
            tag_counts: Mapping of tag string to frequency
            top_k: Maximum number of tags to keep (sorted by frequency)
        """
        sorted_tags = sorted(
            [t for t, c in tag_counts.items() if c >= self.min_frequency],
            key=lambda x: (-tag_counts[x], x)
        )

        if top_k is not None and top_k > 0:
            sorted_tags = sorted_tags[:top_k]

        self.tag_to_index = {self.pad_token: 0, self.unk_token: 1}
        self.index_to_tag = {0: self.pad_token, 1: self.unk_token}
        self.unk_index = 1

        for idx, tag in enumerate(sorted_tags, start=2):
            self.tag_to_index[tag] = idx
            self.index_to_tag[idx] = tag
            self.tag_frequencies[tag] = tag_counts[tag]

        self.ignored_tag_indices = [
            self.tag_to_index[tag] for tag in self.ignored_tags
            if tag in self.tag_to_index
        ]

        self.tags = sorted_tags

        logger.info(f"Vocabulary built with {len(self.tag_to_index)} tags (incl. special tokens)")
    
    def save_vocabulary(self, vocab_path: Path) -> None:
        """Save the vocabulary to a JSON file."""
        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure type consistency: tag_to_index has string keys and int values
        # index_to_tag has string keys (will be converted back to int on load) and string values
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'tag_to_index': {str(k): int(v) for k, v in self.tag_to_index.items()},
                'index_to_tag': {str(k): str(v) for k, v in self.index_to_tag.items()},
                'tag_frequencies': {str(k): int(v) if isinstance(v, (int, float)) else v
                                   for k, v in self.tag_frequencies.items()},
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved vocabulary to {vocab_path}")
    
    def load_vocabulary(self, vocab_path: Path, skip_validation: bool = False) -> None:
        """Load vocabulary from a JSON file with type conversion and validation."""
        # Use orjson if available (3-5x faster)
        if HAS_ORJSON:
            data = orjson.loads(Path(vocab_path).read_bytes())
        else:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Type conversion with full validation (all-or-nothing to prevent partial corruption)
        # Build temporary dicts first, then assign all at once after successful validation
        tag_to_index_raw = data['tag_to_index']
        index_to_tag_raw = data['index_to_tag']

        # Validate and convert tag_to_index: all values must be convertible to int
        # Use temporary dict to prevent partial state on error
        _tag_to_index = {}
        for k, v in tag_to_index_raw.items():
            try:
                _tag_to_index[str(k)] = int(v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid index value for tag '{k}': {v!r} (must be integer)") from e

        # Validate and convert index_to_tag: all keys must be convertible to int
        # Use temporary dict to prevent partial state on error
        _index_to_tag = {}
        for k, v in index_to_tag_raw.items():
            try:
                _index_to_tag[int(k)] = str(v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid index key '{k}' (must be integer)") from e

        # All validation passed - now assign to self (all-or-nothing)
        self.tag_to_index = _tag_to_index
        self.index_to_tag = _index_to_tag
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
        # Note: Some ignored tags may not be in vocabulary
        self.ignored_tag_indices = [
            self.tag_to_index[tag] for tag in self.ignored_tags
            if tag in self.tag_to_index
        ]

        # Validate vocabulary integrity unless skipped (for trusted/cached sources)
        if not skip_validation:
            _verify_vocabulary_integrity(self, vocab_path)

        logger.info(f"Loaded vocabulary with {len(self.tag_to_index)} tags from {vocab_path}")

    def to_json(self) -> str:
        """Serialize vocabulary to JSON string."""
        return json.dumps({
            'tag_to_index': self.tag_to_index,
            'index_to_tag': self.index_to_tag,
            'tag_frequencies': self.tag_frequencies,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'TagVocabulary':
        """Create vocabulary from a JSON string.

        Raises:
            ValueError: If json_str is empty or contains only whitespace
            json.JSONDecodeError: If json_str is not valid JSON
        """
        if not json_str or not json_str.strip():
            detail = "empty" if not json_str else f"whitespace-only (length {len(json_str)})"
            raise ValueError(f"Cannot create vocabulary from {detail} JSON string")
        # Use orjson if available (3-5x faster)
        data = orjson.loads(json_str) if HAS_ORJSON else json.loads(json_str)
        vocab = cls()
        # Validate and convert tag_to_index values to int
        try:
            vocab.tag_to_index = {k: int(v) for k, v in data['tag_to_index'].items()}
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid index value in tag_to_index: {e}") from e
        # Validate and convert index_to_tag keys to int
        try:
            vocab.index_to_tag = {int(k): v for k, v in data['index_to_tag'].items()}
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid index key in index_to_tag: {e}") from e
        vocab.tag_frequencies = data.get('tag_frequencies', {})

        # Ensure special tokens exist with standard indices (PAD=0, UNK=1)
        # This is critical for correct encoding/decoding
        if vocab.pad_token not in vocab.tag_to_index:
            vocab.tag_to_index[vocab.pad_token] = 0
            vocab.index_to_tag[0] = vocab.pad_token
        if vocab.unk_token not in vocab.tag_to_index:
            vocab.tag_to_index[vocab.unk_token] = 1
            vocab.index_to_tag[1] = vocab.unk_token

        # UNK index must match the actual index in tag_to_index (no hardcoded fallback)
        vocab.unk_index = vocab.tag_to_index[vocab.unk_token]
        vocab.tags = [t for t in vocab.tag_to_index.keys() if t not in (vocab.pad_token, vocab.unk_token)]
        vocab.ignored_tag_indices = [
            vocab.tag_to_index[t] for t in vocab.ignored_tags
            if t in vocab.tag_to_index
        ]

        _verify_vocabulary_integrity(vocab, Path('embedded'))
        return vocab
    
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


def load_vocabulary_for_training(vocab_dir: Path = VOCAB_PATH, use_cache: bool = True) -> TagVocabulary:
    """Load vocabulary from directory or file with caching support.

    First tries to load from vocabulary.json, then tags.txt. Uses an in-memory
    cache to avoid reloading the same vocabulary repeatedly in a single process.

    Args:
        vocab_dir: Directory containing vocabulary files OR direct path to vocabulary file
        use_cache: If True, use cached vocabulary if available (default: True)

    Returns:
        TagVocabulary instance

    Raises:
        FileNotFoundError: If vocabulary file not found
        ValueError: If vocabulary is invalid
    """
    vocab_path = Path(vocab_dir).resolve()

    # Check cache first if enabled (thread-safe with double-checked locking)
    cache_key = str(vocab_path)
    if use_cache:
        with _VOCAB_CACHE_LOCK:
            if cache_key in _VOCAB_CACHE:
                logger.debug(f"Using cached vocabulary from {vocab_path}")
                return _VOCAB_CACHE[cache_key]

    vocab = None
    source_path = None

    try:
        # Check if direct file path
        if vocab_path.is_file():
            if vocab_path.suffix == '.json':
                # Load JSON vocabulary directly
                vocab = TagVocabulary()
                vocab.load_vocabulary(vocab_path)
                source_path = vocab_path
            elif vocab_path.suffix == '.txt':
                vocab = TagVocabulary.from_file(vocab_path)
                source_path = vocab_path
            else:
                raise ValueError(f"Unsupported vocabulary file format: {vocab_path.suffix}")

        # Otherwise treat as directory
        elif vocab_path.is_dir():
            # Try JSON format first
            vocab_json = vocab_path / "vocabulary.json"
            if vocab_json.exists():
                vocab = TagVocabulary()
                vocab.load_vocabulary(vocab_json)
                source_path = vocab_json
            else:
                # Try text format
                vocab_file = vocab_path / "tags.txt"
                if vocab_file.exists():
                    vocab = TagVocabulary.from_file(vocab_file)
                    source_path = vocab_file

        # Validate result
        if vocab is None or source_path is None:
            raise FileNotFoundError(
                f"Vocabulary not found at {vocab_path}. "
                f"Tried: vocabulary.json and tags.txt"
            )

        # Verify vocabulary integrity
        _verify_vocabulary_integrity(vocab, source_path)

        # Cache the loaded vocabulary (thread-safe)
        if use_cache:
            with _VOCAB_CACHE_LOCK:
                _VOCAB_CACHE[cache_key] = vocab
            logger.debug(f"Cached vocabulary from {vocab_path}")

        return vocab

    except FileNotFoundError:
        # Re-raise file not found errors directly
        raise
    except Exception as e:
        # Wrap other errors with context
        logger.error(f"Failed to load vocabulary from {vocab_path}: {e}")
        raise ValueError(f"Failed to load vocabulary from {vocab_path}: {e}") from e


def verify_vocabulary_integrity(vocab: TagVocabulary, source_path: Optional[Path] = None,
                               max_placeholders: int = 10) -> None:
    """Verify that a loaded vocabulary contains real tags and has consistent bidirectional mappings.

    Args:
        vocab: TagVocabulary instance to verify
        source_path: Optional path to vocabulary source for error messages
        max_placeholders: Maximum allowed placeholder tags (default 10 for special tokens)

    Raises:
        ValueError: If vocabulary contains too many placeholder tags or inconsistent mappings
    """
    # Check for placeholder tags
    placeholder_count = sum(
        1 for tag in vocab.tag_to_index.keys()
        if tag.startswith("tag_") and len(tag) > 4 and tag[4:].isdigit()
    )

    if placeholder_count > max_placeholders:
        source_str = f" at {source_path}" if source_path else ""
        sample_placeholders = [
            tag for tag in list(vocab.tag_to_index.keys())[:50]
            if tag.startswith("tag_") and len(tag) > 4 and tag[4:].isdigit()
        ][:5]
        raise ValueError(
            f"CRITICAL: Vocabulary{source_str} contains {placeholder_count} placeholder tags "
            f"(max allowed: {max_placeholders}).\n"
            f"Examples: {sample_placeholders}\n"
            f"This vocabulary file is corrupted and contains 'tag_XXX' instead of real tags.\n"
            f"The model would learn meaningless labels if used in training or inference.\n"
            f"Please regenerate the vocabulary from your dataset annotations."
        )

    # Verify bidirectional mapping consistency (addresses CR-004)
    source_str = f" at {source_path}" if source_path else ""

    # Check all indices in tag_to_index map back correctly
    for tag, idx in vocab.tag_to_index.items():
        if not isinstance(idx, int):
            raise ValueError(
                f"CRITICAL: Vocabulary{source_str} has non-integer index for tag '{tag}': {idx} (type: {type(idx).__name__})"
            )
        if idx not in vocab.index_to_tag:
            raise ValueError(
                f"CRITICAL: Vocabulary{source_str} inconsistent - "
                f"tag_to_index['{tag}'] = {idx}, but {idx} not in index_to_tag"
            )
        if vocab.index_to_tag[idx] != tag:
            raise ValueError(
                f"CRITICAL: Vocabulary{source_str} inconsistent - "
                f"tag_to_index['{tag}'] = {idx}, but index_to_tag[{idx}] = '{vocab.index_to_tag[idx]}'"
            )

    # Check all indices in index_to_tag map back correctly
    for idx, tag in vocab.index_to_tag.items():
        if not isinstance(idx, int):
            raise ValueError(
                f"CRITICAL: Vocabulary{source_str} has non-integer key in index_to_tag: {idx} (type: {type(idx).__name__})"
            )
        if tag not in vocab.tag_to_index:
            raise ValueError(
                f"CRITICAL: Vocabulary{source_str} inconsistent - "
                f"index_to_tag[{idx}] = '{tag}', but '{tag}' not in tag_to_index"
            )
        if vocab.tag_to_index[tag] != idx:
            raise ValueError(
                f"CRITICAL: Vocabulary{source_str} inconsistent - "
                f"index_to_tag[{idx}] = '{tag}', but tag_to_index['{tag}'] = {vocab.tag_to_index[tag]}"
            )

# Keep backward compatibility alias
_verify_vocabulary_integrity = verify_vocabulary_integrity

def create_dataset_config(vocab: TagVocabulary) -> Dict:
    """Create a configuration dictionary for dataset initialization.
    
    This ensures ignored indices are properly passed to datasets when using
    multiprocessing with DataLoader.
    
    Args:
        vocab: TagVocabulary instance
        
    Returns:
        Dictionary with configuration including ignored_indices
    """
    return {
        'vocabulary': vocab,
        'ignored_indices': vocab.get_ignored_indices(),
        # Add other dataset configuration parameters as needed
    }

def create_vocabulary_from_datasets(
    dataset_path: Optional[List[Union[str, Path]]] = None,
    *,
    min_frequency: int = 50,
    top_k: int = 100_000,
    num_workers: int = 16,
    chunk_size: int = 2000,
):
    """Create vocabulary from datasets (for training).

    Scans the dataset directories for JSON annotation files, builds a
    frequency‑sorted vocabulary, and saves it to ``vocabulary.json``.

    Args:
        dataset_path: List with a single root directory to scan recursively for ``*.json``
        min_frequency: Minimum tag frequency to include in the vocabulary
        top_k: Maximum number of tags to keep (most frequent first)
        num_workers: Number of worker processes to use for tag counting
        chunk_size: Number of files per worker chunk
    """
    if not dataset_path:
        raise ValueError("dataset_path is required and must contain at least one path")

    json_files: List[str] = []
    data_dir = Path(dataset_path[0])
    # Scan subdirectories only (skip root-level files like train.json, val.json)
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.rglob('*.json'):
                json_files.append(str(file))

    vocab = TagVocabulary(min_frequency=min_frequency)
    num_workers = int(num_workers or 0)
    chunk_size = max(1, int(chunk_size or 0))

    if num_workers > 1 and json_files:
        logger.info(
            f"Building vocabulary from {len(json_files)} files using {num_workers} workers"
        )
        tag_counts: Counter = Counter()
        chunks = _iter_chunks(json_files, chunk_size)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for partial in executor.map(
                _count_tags_in_files,
                chunks,
                itertools.repeat(vocab.ignored_tags),
            ):
                tag_counts.update(partial)
        vocab.build_from_tag_counts(tag_counts, top_k=top_k)
    else:
        vocab.build_from_annotations([Path(f) for f in json_files], top_k=top_k)

    vocab.save_vocabulary(VOCAB_PATH)
    logger.info(f"Created vocabulary with {len(vocab)} tags at {VOCAB_PATH}")

    return vocab


def main():
    # Main function for command-line usage
    import argparse
    parser = argparse.ArgumentParser(description='Generate vocabulary from anime image dataset')
    parser.add_argument('dataset_paths', nargs='+', help='Paths to dataset directories or images')
    args = parser.parse_args()
    create_vocabulary_from_datasets(args.dataset_paths)
    vocab = load_vocabulary_for_training(VOCAB_PATH)
    logger.info(f"Vocabulary size: {len(vocab)} tags")
    logger.info("Vocabulary preparation complete!")


if __name__ == "__main__":
    main()
