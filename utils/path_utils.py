from __future__ import annotations
from pathlib import Path
import re

DEFAULT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

class PathTraversalError(ValueError):
    """Raised when a candidate path escapes the allowed root."""

def sanitize_identifier(s: str, pattern: re.Pattern[str] = DEFAULT_ID_RE) -> str:
    if not pattern.fullmatch(s):
        raise ValueError(f"Invalid identifier: {s!r}. Use only A–Z, a–z, 0–9, _ . -")
    return s


def safe_join(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = (root.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(root)   # Py 3.9+
    except ValueError:
        raise PathTraversalError(f"Path escapes dataset root: {candidate}")
    return candidate


def validate_image_path(
    root: Path,
    stem: str,
    allowed_exts=(".jpg", ".jpeg", ".png", ".webp"),
) -> Path:
    stem = sanitize_identifier(Path(stem).stem)
    # direct matches
    for ext in allowed_exts:
        p = safe_join(root, f"{stem}{ext}")
        if p.exists():
            return p
    # case-insensitive fallback
    try:
        lower_map = {f.name.lower(): f for f in root.iterdir() if f.is_file()}
        for ext in allowed_exts:
            alt = lower_map.get(f"{stem}{ext}".lower())
            if alt:
                return safe_join(root, alt.name)
    except FileNotFoundError:
        pass
    raise FileNotFoundError(f"Image not found under {root}: {stem}")


def resolve_and_confine(root: Path, candidate: str | Path) -> Path:
    """
    Spec-named helper: resolve candidate under root and ensure it does not escape.
    Mirrors safe_join semantics to satisfy 'resolve_and_confine(root, candidate)' deliverable.
    """
    return safe_join(Path(root), str(candidate))
