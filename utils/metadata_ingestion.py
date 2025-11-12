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
from typing import Iterable, List

def parse_tags_field(tags_field) -> List[str]:
    """Turn a raw tags field into a list of tokens.

    Accepts None, string or iterable.  String values are split on
    commas, with whitespace stripped from each tag.  Empty or unsupported
    types yield an empty list.
    """
    if tags_field is None:
        return []
    if isinstance(tags_field, str):
        return [p.strip() for p in tags_field.split(',') if p.strip()]
    if isinstance(tags_field, (list, tuple)):
        return [str(t).strip() for t in tags_field if str(t).strip()]
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
    """Join root and rel, raising if the result escapes the dataset root."""
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise ValueError(f"Path escapes dataset root: {rel}")
    return p

def validate_image_path(root: Path, name: str,
                        allowed_exts=(".jpg", ".jpeg", ".png", ".webp")) -> Path:
    """Resolve and validate an image path.

    Ensures the extension is allowed and the file exists.  Performs a
    case‑insensitive lookup if the exact file is missing.
    """
    p = safe_join(root, name)
    if p.suffix.lower() not in [ext.lower() for ext in allowed_exts]:
        raise ValueError(f"Unsupported image extension: {p.suffix}")
    if p.exists():
        return p
    # case‑insensitive search
    try:
        candidates = {f.name.lower(): f for f in p.parent.iterdir() if f.is_file()}
    except FileNotFoundError:
        candidates = {}
    alt = candidates.get(p.name.lower())
    if alt and alt.exists():
        return alt
    raise FileNotFoundError(f"Image not found: {p}")
