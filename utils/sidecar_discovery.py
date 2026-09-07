"""Shared sidecar discovery for a root containing multiple dataset subsets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DISCOVERY_VERSION = "v2-sidecars-1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif"}
SKIP_DIRS = {"__pycache__", "node_modules", "venv", "env"}


def _include_dir(name: str) -> bool:
    return not name.startswith(".") and name not in SKIP_DIRS


def subset_signature(root: Path) -> str:
    """Cheap detection of new/removed top-level subsets; not a content hash."""
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and _include_dir(p.name))
    return hashlib.sha256((DISCOVERY_VERSION + "\n" + "\n".join(names)).encode()).hexdigest()


def discover_sidecars(root: Path) -> list[Path]:
    """Find JSON/image pairs, ignoring updater state and environment files.

    A directory listing supplies the usual same-stem match without millions of
    individual stat calls. Nonmatching JSONs need a real filename/tags record.
    Symlinks are not followed. Dataset contents are never modified here.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    found = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if _include_dir(d))
        images = {name for name in files if Path(name).suffix.lower() in IMAGE_SUFFIXES}
        if not images:
            continue
        stems = {Path(name).stem for name in images}
        for name in files:
            if not name.lower().endswith(".json") or name in {"train.json", "val.json"}:
                continue
            path = Path(directory) / name
            if path.stem in stems:
                found.append(path)
                continue
            try:
                data = json.loads(path.read_bytes())
                if (isinstance(data, dict) and "tags" in data
                        and data.get("filename") in images):
                    found.append(path)
            except (OSError, ValueError, TypeError):
                pass
    return sorted(found)
