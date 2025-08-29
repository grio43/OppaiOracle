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


def load_vocab(path: str) -> List[str]:
    """
    Load a vocabulary from a JSON file.  The vocabulary is expected to be a
    list of tag strings.  If the file does not exist, an empty list is
    returned.

    Args:
        path: Path to the vocabulary JSON file.
    Returns:
        Ordered list of tag names.
    """
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    # Accept either a list or a dict mapping tag to index.  In the latter case
    # reconstruct the list by sorting by index.
    if isinstance(data, list):
        return list(data)
    elif isinstance(data, dict):
        # assume {tag: index}
        return [tag for tag, _ in sorted(data.items(), key=lambda kv: kv[1])]
    else:
        raise ValueError(f"Unsupported vocabulary format in {path}: {type(data)}")


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
