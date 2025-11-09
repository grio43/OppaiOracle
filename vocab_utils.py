"""
Utilities for managing the vocabulary (tag list) used by OppaiOracle.

The vocabulary strategy described in the runbook treats the label vocabulary as a
versioned artifact.  This module provides helpers to load, save and compare
vocabularies as well as compute cryptographic hashes over ordered tag lists.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict


def load_vocab(path: str, allow_missing: bool = True) -> List[str]:
    """
    Load a vocabulary from a JSON file.  The vocabulary is expected to be a
    list of tag strings.

    Args:
        path: Path to the vocabulary JSON file.
        allow_missing: If True, return empty list for missing files;
                      if False, raise FileNotFoundError
    Returns:
        Ordered list of tag names.
    Raises:
        FileNotFoundError: If file doesn't exist and allow_missing=False
        ValueError: If JSON is invalid, encoding error, or unsupported format
        PermissionError: If file cannot be read due to permissions
        OSError: If file cannot be read due to I/O error
    """
    p = Path(path)
    if not p.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    try:
        content = p.read_text(encoding='utf-8')
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading vocabulary file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"I/O error reading vocabulary file {path}: {e}") from e

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in vocabulary file {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error in vocabulary file {path}: {e}. Expected UTF-8.") from e

    # Accept either a list or a dict mapping tag to index.  In the latter case
    # reconstruct the list by sorting by index.
    if isinstance(data, list):
        return list(data)
    elif isinstance(data, dict):
        # assume {tag: index}
        try:
            return [tag for tag, _ in sorted(data.items(), key=lambda kv: kv[1])]
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid vocabulary dict structure in {path}: {e}")
    else:
        raise ValueError(f"Unsupported vocabulary format in {path}: expected list or dict, got {type(data)}")


def save_vocab(vocab: List[str], path: str) -> None:
    """
    Save a vocabulary to a JSON file.  Writes the ordered list of tags.

    Args:
        vocab: Ordered list of tag names.
        path: Destination file path.
    """
    p = Path(path)
    p.write_text(json.dumps(list(vocab), ensure_ascii=False, indent=2))


def compute_vocab_hash(vocab: List[str]) -> str:
    """
    Compute a sha256 hash over the ordered tag list.  The hash is computed over
    the newline-separated tag names to ensure ordering affects the digest.

    Args:
        vocab: Ordered list of tag names.
    Returns:
        Hexadecimal sha256 digest.
    """
    joined = "\n".join(vocab).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def diff_vocab(old_vocab: List[str], new_vocab: List[str]) -> Dict[str, List[str]]:
    """
    Compute a diff between two vocabularies.  Returns a dict containing added
    and removed tags.  Renamed tags cannot be detected reliably; rename
    detection should be implemented via an alias map.

    Args:
        old_vocab: The canonical vocabulary loaded from vocabulary.json.
        new_vocab: A vocabulary derived from current training data.
    Returns:
        Dict with keys 'added' and 'removed', each mapping to a list of tag names.
    """
    old_set = set(old_vocab)
    new_set = set(new_vocab)
    added = [tag for tag in new_vocab if tag not in old_set]
    removed = [tag for tag in old_vocab if tag not in new_set]
    return {"added": added, "removed": removed}
