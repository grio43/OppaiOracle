#!/usr/bin/env python3
"""
Utility functions for metadata ingestion with proper tag handling
"""
from pathlib import Path
from typing import List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def parse_tags_field(tags_field: Any) -> List[str]:
    """
    Parse tags field from various formats to a list of strings.
    Handles single space-delimited strings, lists, and None.
    
    Args:
        tags_field: Tags in various formats (str, list, or None)
        
    Returns:
        List of tag strings
    """
    if tags_field is None:
        return []
    if isinstance(tags_field, str):
        # Split on whitespace only; DO NOT modify punctuation/case
        parts = [p for p in tags_field.split() if p]
        return parts
    if isinstance(tags_field, (list, tuple)):
        return [str(t) for t in tags_field if str(t)]
    return []


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """
    Remove duplicates from list while preserving order.
    First occurrence of each item is kept.
    
    Args:
        items: List with potential duplicates
        
    Returns:
        Deduplicated list with order preserved
    """
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_join(root: Path, rel: str) -> Path:
    """
    Safely join paths preventing directory traversal.
    
    Args:
        root: Root directory path
        rel: Relative path to join
        
    Returns:
        Joined path
        
    Raises:
        ValueError: If path would escape root directory
    """
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise ValueError(f"Path escapes dataset root: {rel}")
    return p


def validate_image_path(root: Path, name: str, 
                        allowed_exts: tuple = (".jpg", ".jpeg", ".png", ".webp")) -> Path:
    """
    Validate and resolve image path with case-insensitive fallback.
    
    Args:
        root: Dataset root directory
        name: Image filename
        allowed_exts: Tuple of allowed extensions
        
    Returns:
        Validated path to image
        
    Raises:
        ValueError: If extension not allowed
        FileNotFoundError: If image not found
    """
    p = safe_join(root, name)
    if p.suffix.lower() not in allowed_exts:
        raise ValueError(f"Unsupported image extension: {p.suffix}")
    
    if not p.exists():
        # Try case-insensitive lookup
        dirp = p.parent
        try:
            candidates = {f.name.lower(): f for f in dirp.iterdir() if f.is_file()}
            alt = candidates.get(p.name.lower())
        except FileNotFoundError:
            alt = None
        
        if alt and alt.exists():
            return alt
        raise FileNotFoundError(f"Image not found: {p}")
    
    return p