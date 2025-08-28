# utils/metadata_ingestion.py

"""
Common utilities for ingesting image metadata files.

Functions here convert a raw “tags” field into a list of tag tokens,
deduplicate tags while preserving order, and perform safe filename
resolution to prevent path traversal.  These helpers perform *no*
normalisation on punctuation or case; upstream code must preserve tags
exactly.
"""

from pathlib import Path
# Reuse canonicalized helpers
from .path_utils import safe_join as _safe_join, validate_image_path as _validate_image_path
from typing import Iterable, List

def parse_tags_field(tags_field) -> List[str]:
    """Turn a raw tags field into a list of tokens.

    Accepts None, string or iterable.  String values are split on
    whitespace only (no punctuation stripping).  Empty or unsupported
    types yield an empty list.
    """
    if tags_field is None:
        return []
    if isinstance(tags_field, str):
        return [p for p in tags_field.split() if p]
    if isinstance(tags_field, (list, tuple)):
        return [str(t) for t in tags_field if str(t)]
    return []

def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Return a list with duplicates removed while preserving order."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def safe_join(root: Path, rel: str) -> Path:
    return _safe_join(root, rel)

def validate_image_path(root: Path, name: str,
                        allowed_exts=(".jpg", ".jpeg", ".png", ".webp")) -> Path:
    return _validate_image_path(root, str(name), allowed_exts)
