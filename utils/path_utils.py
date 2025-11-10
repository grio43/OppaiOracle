from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import re

DEFAULT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

class PathTraversalError(ValueError):
    """Raised when a candidate path escapes the allowed root."""

def sanitize_identifier(s: str, pattern: re.Pattern[str] = DEFAULT_ID_RE) -> str:
    """Sanitize an identifier to prevent path traversal attacks.

    Args:
        s: The identifier to sanitize
        pattern: Regex pattern for allowlist validation (default: alphanumeric, _, ., -)

    Returns:
        The sanitized identifier

    Raises:
        ValueError: If identifier contains disallowed characters or path traversal sequences
    """
    if not pattern.fullmatch(s):
        raise ValueError(f"Invalid identifier: {s!r}. Use only A–Z, a–z, 0–9, _ . -")

    # Additional security checks for path traversal
    if '..' in s:
        raise ValueError(f"Path traversal sequence '..' not allowed in identifier: {s!r}")
    if s.startswith('.') or s.startswith('-'):
        raise ValueError(f"Identifier cannot start with '.' or '-': {s!r}")
    if '/' in s or '\\' in s:
        raise ValueError(f"Path separators not allowed in identifier: {s!r}")

    # Verify no null bytes (can cause issues in C-based filesystem calls)
    if '\x00' in s:
        raise ValueError(f"Null bytes not allowed in identifier: {s!r}")

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
    allowed_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp"),
    *,
    allowed_external_roots: Optional[Sequence[Path]] = None,
) -> Path:
    """
    Find an image by sanitized stem under a directory, allowing symlink targets
    to live under additional trusted roots.

    - root: directory that holds the expected image file (e.g., <dataset>/images)
    - stem: filename without extension; only [A-Za-z0-9_.-] allowed
    - allowed_external_roots: optional additional roots within which a symlink
      target may resolve. If None, targets must resolve under 'root'.
    """
    stem = sanitize_identifier(Path(stem).stem)
    roots: list[Path] = [root.resolve()]
    if allowed_external_roots:
        roots.extend([Path(r).resolve() for r in allowed_external_roots])

    def _within_any(p: Path) -> bool:
        pr = p.resolve()
        for r in roots:
            try:
                pr.relative_to(r)
                return True
            except ValueError:
                continue
        return False

    def _check_and_return(p: Path) -> Optional[Path]:
        if not p.exists():
            return None
        # If it's a symlink (or any path), ensure its resolved target is within
        # at least one trusted root.
        try:
            target = p.resolve()
        except FileNotFoundError:
            return None
        if not _within_any(target):
            raise PathTraversalError(f"Path escapes dataset root: {target}")
        return p

    # direct matches
    for ext in allowed_exts:
        cand = root / f"{stem}{ext}"
        out = _check_and_return(cand)
        if out is not None:
            return out
    # case-insensitive fallback
    try:
        entries = list(root.iterdir())
        lower_map = {f.name.lower(): f for f in entries if f.is_file() or f.is_symlink()}
        for ext in allowed_exts:
            alt = lower_map.get(f"{stem}{ext}".lower())
            if alt:
                out = _check_and_return(root / alt.name)
                if out is not None:
                    return out
    except FileNotFoundError:
        pass
    raise FileNotFoundError(f"Image not found under {root}: {stem}")


def resolve_and_confine(root: Path, candidate: str | Path) -> Path:
    """
    Spec-named helper: resolve candidate under root and ensure it does not escape.
    Mirrors safe_join semantics to satisfy 'resolve_and_confine(root, candidate)' deliverable.
    """
    return safe_join(Path(root), str(candidate))
