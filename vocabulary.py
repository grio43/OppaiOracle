#!/usr/bin/env python3
"""
Shared vocabulary module for anime image tagger.

This module provides a unified TagVocabulary class that handles tag vocabulary
management for both training and inference, eliminating code duplication.
"""

import atexit
import copy
import hashlib
import json
import logging
import itertools
import os
import random
import tempfile
import threading
import time

# Try to use orjson for faster JSON parsing (3-5x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# Define JSON decode error types for exception handling (orjson has its own error type)
_JSON_ERRORS = (json.JSONDecodeError, orjson.JSONDecodeError) if HAS_ORJSON else (json.JSONDecodeError,)

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union
import torch
import yaml

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
VOCAB_PATH = PROJECT_ROOT / "vocabulary.json"
VOCAB_CACHE_DIR = PROJECT_ROOT / "logs" / "vocab_cache"

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


def _compute_dataset_hash(dataset_path: Path) -> str:
    """Compute a stable hash for the dataset path."""
    return hashlib.sha1(str(dataset_path.resolve()).encode()).hexdigest()[:16]


def _validate_file_list_cache(cached_files: List[str], sample_size: int = 1000) -> bool:
    """Validate cached file list by checking a stratified sample exists.

    Args:
        cached_files: List of cached file paths
        sample_size: Number of files to sample for validation

    Returns:
        True if cache appears valid, False otherwise
    """
    if not cached_files:
        return False

    # Sample files from beginning, middle, and end for better coverage
    total = len(cached_files)
    if total <= sample_size:
        sample = cached_files
    else:
        # Stratified sampling: beginning, middle, end
        third = sample_size // 3
        sample = (
            cached_files[:third] +
            cached_files[total // 2 - third // 2 : total // 2 + third // 2] +
            cached_files[-third:]
        )

    # Check existence in parallel using ThreadPoolExecutor
    def check_exists(path: str) -> bool:
        return os.path.exists(path)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(check_exists, sample))

    # Allow 1% tolerance for deleted files
    valid_count = sum(results)
    return valid_count >= len(sample) * 0.99


def _get_cached_file_list(
    dataset_path: Path,
    use_cache: bool = True
) -> List[str]:
    """Get file list from cache or discover via rglob.

    Args:
        dataset_path: Root directory to scan for JSON files
        use_cache: Whether to use/create cache

    Returns:
        List of JSON file paths as strings
    """
    data_dir = Path(dataset_path).resolve()
    cache_key = _compute_dataset_hash(data_dir)
    cache_file = VOCAB_CACHE_DIR / f"{cache_key}.filelist.txt"

    # Try loading from cache
    if use_cache and cache_file.exists():
        try:
            start = time.perf_counter()
            with open(cache_file, 'r', encoding='utf-8') as f:
                # Skip header line
                header = f.readline().strip()
                if header.startswith("# vocab_cache v1"):
                    cached_files = [line.strip() for line in f if line.strip()]

                    # Validate cache
                    if _validate_file_list_cache(cached_files):
                        elapsed = time.perf_counter() - start
                        logger.info(
                            f"Loaded {len(cached_files):,} files from cache in {elapsed:.2f}s"
                        )
                        return cached_files
                    else:
                        logger.warning("File list cache validation failed, rescanning...")
        except Exception as e:
            logger.warning(f"Failed to load file list cache: {e}")

    # Discover files via parallel directory scanning
    start = time.perf_counter()
    json_files = _parallel_scan_directories(data_dir)
    elapsed = time.perf_counter() - start
    logger.info(f"Discovered {len(json_files):,} JSON files in {elapsed:.2f}s")

    # Save to cache (atomic write to prevent corruption on crash)
    if use_cache and json_files:
        try:
            VOCAB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Write to temp file first, then atomically rename
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=VOCAB_CACHE_DIR,
                delete=False, suffix='.tmp'
            ) as tmp:
                tmp.write(f"# vocab_cache v1 | count={len(json_files)} | path={data_dir}\n")
                for path in json_files:
                    tmp.write(f"{path}\n")
                tmp_path = tmp.name
            os.replace(tmp_path, cache_file)  # Atomic on POSIX and Windows
            logger.info(f"Cached file list to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache file list: {e}")
            # Clean up temp file if it exists
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    return json_files


def _parallel_scan_directories(root: Path, num_workers: int = 16) -> List[str]:
    """Scan directories in parallel for JSON files.

    Uses os.scandir() which is faster than Path.iterdir() and scans
    subdirectories in parallel using ThreadPoolExecutor.

    Args:
        root: Root directory to scan
        num_workers: Number of parallel workers

    Returns:
        List of JSON file paths as strings
    """
    json_files: List[str] = []
    dirs_to_scan: List[Path] = []

    # First pass: collect subdirectories from root
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    dirs_to_scan.append(Path(entry.path))
    except Exception as e:
        logger.error(f"Failed to scan root directory {root}: {e}")
        return json_files

    if not dirs_to_scan:
        # No subdirectories, scan root directly
        return _scan_single_directory(root)

    def scan_subdir(subdir: Path) -> List[str]:
        """Recursively scan a subdirectory for JSON files."""
        files = []
        try:
            for file in subdir.rglob("*.json"):
                files.append(str(file))
        except Exception as e:
            logger.warning(f"Error scanning {subdir}: {e}")
        return files

    # Parallel scan of subdirectories
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scan_subdir, d): d for d in dirs_to_scan}
        for future in as_completed(futures):
            try:
                json_files.extend(future.result())
            except Exception as e:
                logger.warning(f"Error in parallel scan: {e}")

    return json_files


def _scan_single_directory(directory: Path) -> List[str]:
    """Scan a single directory recursively for JSON files."""
    files = []
    try:
        for file in directory.rglob("*.json"):
            files.append(str(file))
    except Exception as e:
        logger.warning(f"Error scanning {directory}: {e}")
    return files


def _get_cached_frequencies(
    dataset_path: Path,
    file_list: List[str],
    use_cache: bool = True
) -> Optional[Counter]:
    """Get tag frequencies from cache if valid.

    Args:
        dataset_path: Dataset root path
        file_list: Current file list (for validation)
        use_cache: Whether to use cache

    Returns:
        Counter of tag frequencies or None if cache miss
    """
    if not use_cache:
        return None

    data_dir = Path(dataset_path).resolve()
    cache_key = _compute_dataset_hash(data_dir)
    cache_file = VOCAB_CACHE_DIR / f"{cache_key}.frequencies.json"

    if not cache_file.exists():
        return None

    try:
        start = time.perf_counter()

        # Load cache
        if HAS_ORJSON:
            data = orjson.loads(cache_file.read_bytes())
        else:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Validate file count matches (within 0.1% tolerance)
        cached_count = data.get('_file_count', 0)
        current_count = len(file_list)
        tolerance = max(10, int(current_count * 0.001))

        if abs(cached_count - current_count) > tolerance:
            logger.info(
                f"Frequency cache stale: {cached_count:,} vs {current_count:,} files"
            )
            return None

        # Extract frequencies (exclude metadata keys)
        frequencies = Counter({
            k: v for k, v in data.items()
            if not k.startswith('_')
        })

        elapsed = time.perf_counter() - start
        logger.info(
            f"Loaded {len(frequencies):,} tag frequencies from cache in {elapsed:.2f}s"
        )
        return frequencies

    except Exception as e:
        logger.warning(f"Failed to load frequency cache: {e}")
        return None


def _save_frequency_cache(
    dataset_path: Path,
    tag_counts: Counter,
    file_count: int
) -> None:
    """Save tag frequencies to cache.

    Args:
        dataset_path: Dataset root path
        tag_counts: Counter of tag frequencies
        file_count: Number of files processed (for validation)
    """
    data_dir = Path(dataset_path).resolve()
    cache_key = _compute_dataset_hash(data_dir)
    cache_file = VOCAB_CACHE_DIR / f"{cache_key}.frequencies.json"

    tmp_path = None
    try:
        VOCAB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Add metadata
        cache_data = dict(tag_counts)
        cache_data['_file_count'] = file_count
        cache_data['_timestamp'] = time.time()
        cache_data['_version'] = 1

        # Write to temp file first, then atomically rename
        if HAS_ORJSON:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=VOCAB_CACHE_DIR, delete=False, suffix='.tmp'
            ) as tmp:
                tmp.write(orjson.dumps(cache_data))
                tmp_path = tmp.name
        else:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=VOCAB_CACHE_DIR,
                delete=False, suffix='.tmp'
            ) as tmp:
                json.dump(cache_data, tmp)
                tmp_path = tmp.name

        os.replace(tmp_path, cache_file)  # Atomic on POSIX and Windows
        logger.info(f"Cached {len(tag_counts):,} tag frequencies to {cache_file}")

    except Exception as e:
        logger.warning(f"Failed to cache frequencies: {e}")
        # Clean up temp file if it exists
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _count_tags_parallel(
    json_files: List[str],
    ignored_tags: Set[str],
    num_workers: int = 20,
    chunk_size: int = 10_000
) -> Counter:
    """Count tags in parallel using ThreadPoolExecutor.

    Uses ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
    IPC pickle overhead. orjson releases the GIL during parsing,
    enabling true parallelism for I/O-bound work.

    Args:
        json_files: List of JSON file paths
        ignored_tags: Set of tags to ignore
        num_workers: Number of parallel workers
        chunk_size: Files per chunk

    Returns:
        Counter of tag frequencies
    """
    if not json_files:
        return Counter()

    start = time.perf_counter()
    tag_counts: Counter = Counter()

    # Process in chunks for better memory management
    chunks = list(_iter_chunks(json_files, chunk_size))
    total_chunks = len(chunks)

    logger.info(
        f"Counting tags in {len(json_files):,} files "
        f"({total_chunks} chunks, {num_workers} workers)"
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_count_tags_in_files, chunk, ignored_tags)
            for chunk in chunks
        ]

        completed = 0
        for future in as_completed(futures):
            try:
                partial_counts = future.result()
                tag_counts.update(partial_counts)
                completed += 1

                if completed % 50 == 0 or completed == total_chunks:
                    logger.info(f"Progress: {completed}/{total_chunks} chunks")
            except Exception as e:
                logger.warning(f"Error in tag counting: {e}")

    elapsed = time.perf_counter() - start
    logger.info(
        f"Counted {sum(tag_counts.values()):,} tag occurrences "
        f"({len(tag_counts):,} unique) in {elapsed:.2f}s"
    )

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
        dtype_key = str(tag_vector_dtype).lower()
        if dtype_key not in dtype_map:
            raise ValueError(
                f"Invalid tag_vector_dtype '{tag_vector_dtype}'. "
                f"Must be one of: {list(dtype_map.keys())}"
            )
        self._tag_vector_dtype = dtype_map[dtype_key]

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

    def copy(self) -> 'TagVocabulary':
        """Create a shallow copy with independent mutable containers.

        This is more efficient than deepcopy while still providing isolation
        for the mutable dict/list attributes that could be modified.

        Returns:
            A new TagVocabulary instance with copied containers
        """
        new_vocab = TagVocabulary.__new__(TagVocabulary)
        # Copy mutable containers to prevent cross-modification
        new_vocab.tag_to_index = self.tag_to_index.copy()
        new_vocab.index_to_tag = self.index_to_tag.copy()
        new_vocab.tag_frequencies = self.tag_frequencies.copy()
        new_vocab.ignored_tags = self.ignored_tags.copy()
        new_vocab.ignored_tag_indices = self.ignored_tag_indices.copy()
        new_vocab.tags = self.tags.copy()
        new_vocab.rating_to_index = self.rating_to_index.copy()
        # Copy immutable/scalar attributes directly
        new_vocab.min_frequency = self.min_frequency
        new_vocab.pad_token = self.pad_token
        new_vocab.unk_token = self.unk_token
        new_vocab.unk_index = self.unk_index
        new_vocab._tag_vector_dtype = self._tag_vector_dtype
        return new_vocab

    def get_ignored_indices(self) -> List[int]:
        """Get the list of ignored tag indices.
        
        This should be passed to dataset constructors when using multiprocessing.
        """
        return self.ignored_tag_indices.copy()
    
    def encode_tags(self, tags: Iterable[str]) -> torch.Tensor:
        """Encode tag strings into a multi-hot tensor of shape (vocab_size,)."""
        vocab_size = len(self.tag_to_index)
        vector = torch.zeros(vocab_size, dtype=self._tag_vector_dtype)
        for tag in tags:
            if tag in self.ignored_tags:
                continue
            idx = self.tag_to_index.get(tag, self.tag_to_index.get(self.unk_token, self.unk_index))
            # Bounds check to prevent IndexError with corrupted vocabularies
            if idx < 0 or idx >= vocab_size:
                logger.warning(f"Tag '{tag}' has out-of-bounds index {idx} (vocab_size={vocab_size}), skipping")
                continue
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

        # Convert frequencies to int for type consistency (matches save behavior)
        raw_freqs = data.get('tag_frequencies', {})
        vocab.tag_frequencies = {
            str(k): int(v) for k, v in raw_freqs.items()
            if isinstance(v, (int, float))
        }

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

        WARNING: This method does NOT populate tag_frequencies. If you need
        frequency data, use load_vocabulary() with a JSON file instead.

        Args:
            filepath: Path to text file with tags

        Returns:
            TagVocabulary instance (with empty tag_frequencies)

        Raises:
            ValueError: If duplicate tags are found in the file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = [line.strip() for line in f if line.strip()]

        # Check for duplicate tags (would corrupt bidirectional mapping)
        seen: Set[str] = set()
        duplicates: List[str] = []
        for tag in tags:
            if tag in seen:
                duplicates.append(tag)
            seen.add(tag)

        if duplicates:
            raise ValueError(
                f"Duplicate tags in {filepath}: {duplicates[:5]}"
                f"{' (and more)' if len(duplicates) > 5 else ''}"
            )

        vocab = cls()

        # Build vocabulary from tags list
        vocab.tag_to_index = {vocab.pad_token: 0, vocab.unk_token: 1}
        vocab.index_to_tag = {0: vocab.pad_token, 1: vocab.unk_token}
        vocab.unk_index = 1

        for idx, tag in enumerate(tags, start=2):
            vocab.tag_to_index[tag] = idx
            vocab.index_to_tag[idx] = tag

        vocab.tags = tags

        # Validate vocabulary integrity
        _verify_vocabulary_integrity(vocab, filepath)

        logger.warning(
            f"Loaded vocabulary from text file {filepath}. "
            "tag_frequencies not available (use JSON format for frequencies)."
        )

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
                # Return a copy to prevent callers from modifying the cached instance
                return _VOCAB_CACHE[cache_key].copy()

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

        # Note: Vocabulary integrity is verified internally by load_vocabulary()
        # and from_file(), so no additional verification needed here.

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
    num_workers: int = 20,
    chunk_size: int = 10_000,
    use_cache: bool = True,
):
    """Create vocabulary from datasets (for training).

    Scans the dataset directories for JSON annotation files, builds a
    frequency-sorted vocabulary, and saves it to ``vocabulary.json``.

    Performance optimizations for large datasets (5M+ files):
    - Parallel directory scanning with ThreadPoolExecutor
    - File list caching (skips rglob on subsequent runs)
    - Tag frequency caching (instant rebuild with different min_frequency/top_k)
    - ThreadPoolExecutor for I/O-bound JSON parsing (avoids IPC overhead)

    Args:
        dataset_path: List with a single root directory to scan recursively for ``*.json``
        min_frequency: Minimum tag frequency to include in the vocabulary
        top_k: Maximum number of tags to keep (most frequent first)
        num_workers: Number of parallel workers for tag counting (default: 20)
        chunk_size: Number of files per chunk (default: 10,000)
        use_cache: Whether to use file list and frequency caching (default: True)
    """
    if not dataset_path:
        raise ValueError("dataset_path is required and must contain at least one path")

    total_start = time.perf_counter()
    data_dir = Path(dataset_path[0])

    # Step 1: Get file list (from cache or parallel scan)
    json_files = _get_cached_file_list(data_dir, use_cache=use_cache)

    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")

    # Step 2: Check for cached tag frequencies
    vocab = TagVocabulary(min_frequency=min_frequency)
    tag_counts = _get_cached_frequencies(data_dir, json_files, use_cache=use_cache)

    # Step 3: Count tags if not cached
    if tag_counts is None:
        num_workers = int(num_workers or 1)
        chunk_size = max(1, int(chunk_size or 10_000))

        tag_counts = _count_tags_parallel(
            json_files,
            vocab.ignored_tags,
            num_workers=num_workers,
            chunk_size=chunk_size
        )

        # Save frequencies to cache for future runs
        if use_cache:
            _save_frequency_cache(data_dir, tag_counts, len(json_files))

    # Step 4: Build vocabulary from frequencies
    vocab.build_from_tag_counts(tag_counts, top_k=top_k)
    vocab.save_vocabulary(VOCAB_PATH)

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        f"Created vocabulary with {len(vocab)} tags at {VOCAB_PATH} "
        f"(total: {total_elapsed:.2f}s)"
    )

    return vocab


def clear_vocabulary_build_cache(dataset_path: Optional[Path] = None) -> None:
    """Clear the vocabulary build cache.

    Args:
        dataset_path: If provided, only clear cache for this dataset.
                     If None, clear all vocabulary build caches.
    """
    if not VOCAB_CACHE_DIR.exists():
        logger.info("No vocabulary cache directory found")
        return

    if dataset_path is not None:
        # Clear cache for specific dataset
        cache_key = _compute_dataset_hash(Path(dataset_path).resolve())
        patterns = [
            VOCAB_CACHE_DIR / f"{cache_key}.filelist.txt",
            VOCAB_CACHE_DIR / f"{cache_key}.frequencies.json",
        ]
        for cache_file in patterns:
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Deleted cache: {cache_file}")
    else:
        # Clear all caches
        import shutil
        shutil.rmtree(VOCAB_CACHE_DIR)
        logger.info(f"Deleted all vocabulary build caches at {VOCAB_CACHE_DIR}")


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
