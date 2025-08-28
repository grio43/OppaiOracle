"""Common path utilities for dataset handling."""

from pathlib import Path


def safe_join(root: Path, rel: str) -> Path:
    """Join root and rel, raising if the result escapes the dataset root."""
    p = (root / rel).resolve()
    root_resolved = root.resolve()
    if not str(p).startswith(str(root_resolved)):
        raise ValueError(f"Path escapes dataset root: {rel}")
    return p


def validate_image_path(root: Path, name: str,
                         allowed_exts=(".jpg", ".jpeg", ".png", ".webp")) -> Path:
    """Resolve and validate an image path from a base name.

    Searches for a file with any of the allowed extensions (case-insensitive)
    relative to ``root``.
    """
    base = Path(name)
    allowed = [ext.lower() for ext in allowed_exts]
    for ext in allowed:
        candidate = safe_join(root, str(base.with_suffix(ext)))
        if candidate.exists():
            return candidate
        # case-insensitive search
        try:
            files = {f.name.lower(): f for f in candidate.parent.iterdir() if f.is_file()}
        except FileNotFoundError:
            continue
        alt = files.get(candidate.name.lower())
        if alt and alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found: {base}")
