#!/usr/bin/env python3
"""
Generate manifest files (train.json, val.json) for OppaiOracle.

This tool scans per-image JSON sidecars under the active dataset root,
maps tags to current vocabulary indices, and writes manifest files that
the loader can consume in "manifest mode". It can also create a flat
images/ directory of symlinks so the loader can resolve image paths
quickly without crawling shard directories.

Manifest entry schema:
  {
    "image_id": "123456789",   # basename without extension
    "labels": [12, 90, 98765],  # integer indices into vocabulary
    "rating": "safe"           # optional; string or int
  }

Notes:
- Orientation-aware flips are disabled in manifest mode by design.
- Labels include <UNK> (index 1) for unknown tags, but never include <PAD> (0).
- The script streams output; it does not hold all entries in memory.
- For very large datasets, creating millions of symlinks in images/ can be
  heavy on some filesystems. Symlinks are the default to avoid copying data.

Usage:
  python tools/generate_manifests.py \
    [--config configs/unified_config.yaml] \
    [--root /path/to/dataset/root] \
    [--val-ratio 0.05] [--seed 42] \
    [--link-mode symlink|copy|none] \
    [--limit 0] [--no-split-cache]

If --root is not provided, the tool reads configs/unified_config.yaml and
uses the first enabled data.storage_locations[*].path.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os

# Try to use orjson for faster JSON parsing (3-5x faster than stdlib json)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Ensure repo root is importable when running as a script from tools/
import sys as _sys
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from Configuration_System import load_config
from utils.metadata_ingestion import parse_tags_field
from utils.path_utils import sanitize_identifier, validate_image_path as _validate_in_dir
from vocabulary import load_vocabulary_for_training, TagVocabulary, verify_vocabulary_integrity


logger = logging.getLogger("manifest_gen")


ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _find_active_data_root(config) -> Path:
    data_cfg = getattr(config, "data", None)
    if not data_cfg:
        raise RuntimeError("Config has no 'data' section")
    storage_locations = getattr(data_cfg, "storage_locations", []) or []
    for loc in storage_locations:
        try:
            enabled = bool(loc.get("enabled", False)) if isinstance(loc, dict) else bool(getattr(loc, "enabled", False))
            path = loc.get("path") if isinstance(loc, dict) else getattr(loc, "path", None)
        except Exception:
            enabled, path = False, None
        if enabled and path:
            return Path(str(path))
    raise RuntimeError("No enabled storage location found in config.data.storage_locations")


def _split_cache_paths(root: Path) -> Tuple[Path, Path]:
    """Return cache file paths used by dataset_loader for fast split reuse."""
    proj_root = Path(__file__).resolve().parents[1]  # repo root
    splits_dir = proj_root / "logs" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return (
        splits_dir / f"{key}.train.txt",
        splits_dir / f"{key}.val.txt",
    )


def _iter_sidecar_jsons(root: Path, use_cache: bool = True) -> Tuple[List[Path], List[Path]]:
    """Return (train_list, val_list) of JSON sidecars to process.

    If a cached split exists under logs/splits, reuse it. Otherwise, scan
    recursively and do a deterministic 95/5 split (seeded later by caller).
    """
    train_file, val_file = _split_cache_paths(root)
    if use_cache and train_file.exists() and val_file.exists():
        try:
            train_list = [Path(line.strip()) for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            val_list = [Path(line.strip()) for line in val_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            # basic sanity: ensure a few exist
            miss = sum(1 for p in train_list[:100] if not p.exists()) + sum(1 for p in val_list[:100] if not p.exists())
            if miss <= 1:
                logger.info("Using cached split lists: train=%d, val=%d", len(train_list), len(val_list))
                return train_list, val_list
        except Exception:
            pass

    # Scan subdirectories only (skip root-level files like train.json, val.json)
    all_jsons = []
    if root.exists():
        for subdir in root.iterdir():
            if subdir.is_dir():
                all_jsons.extend(subdir.rglob("*.json"))
        all_jsons.sort()
    if not all_jsons:
        raise FileNotFoundError(f"No annotation JSON files found under {root}")
    return all_jsons, []  # caller will split


def _find_image_for_json(json_path: Path, data: dict) -> Tuple[Path, str, str]:
    """Locate the image file for a sidecar JSON.

    Returns: (source_path, image_id_stem, extension)
    """
    # Prefer explicit filename from JSON when present
    fname = data.get("filename")
    if isinstance(fname, str) and fname:
        # Ensure we only use the basename; strip any path components
        cand = Path(fname).name
        stem = sanitize_identifier(Path(cand).stem)
        # Search in the JSON's directory
        src = None
        try:
            src = _validate_in_dir(json_path.parent, f"{stem}{Path(cand).suffix}")
        except Exception:
            # Try allowed extensions by stem
            for ext in ALLOWED_EXTS:
                try:
                    src = _validate_in_dir(json_path.parent, f"{stem}{ext}")
                    break
                except Exception:
                    continue
        if src and src.exists():
            return src, stem, src.suffix.lower()

    # Fallback: use JSON basename stem and try common extensions
    stem = sanitize_identifier(json_path.stem)
    for ext in ALLOWED_EXTS:
        try:
            src = _validate_in_dir(json_path.parent, f"{stem}{ext}")
            return src, stem, src.suffix.lower()
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find image for {json_path}")


def _labels_for_tags(tags: Iterable[str], vocab: TagVocabulary) -> List[int]:
    out_set = set()
    unk_idx = vocab.tag_to_index.get(vocab.unk_token, getattr(vocab, "unk_index", 1))
    for t in tags:
        if t in vocab.ignored_tags:
            continue
        idx = vocab.tag_to_index.get(t, unk_idx)
        if idx != 0:  # never include PAD
            out_set.add(int(idx))
    return sorted(out_set)


def _make_unique_image_id(stem: str, ext: str, images_dir: Path, target: Path) -> Tuple[str, Path]:
    """Ensure a unique image_id within images_dir; return (image_id, link_path)."""
    base = sanitize_identifier(stem)
    link = images_dir / f"{base}{ext}"
    if not link.exists():
        return base, link
    # If exists and points to same file, reuse
    try:
        if link.is_symlink() and link.resolve() == target.resolve():
            return base, link
    except Exception:
        pass
    # Otherwise, disambiguate using a short hash of the target path
    h = hashlib.sha1(str(target).encode("utf-8")).hexdigest()[:8]
    alt = f"{base}-{h}"
    return alt, images_dir / f"{alt}{ext}"


def _ensure_link(target: Path, link: Path, mode: str = "symlink") -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if mode == "none":
        return
    if link.exists():
        # If an existing file/symlink points elsewhere and mode is copy/symlink,
        # we leave it as-is to avoid churn.
        return
    if mode == "symlink":
        os.symlink(str(target), str(link))
    elif mode == "copy":
        # Shallow copy; for huge datasets this will be expensive and is not recommended.
        import shutil as _sh
        _sh.copy2(str(target), str(link))
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _stream_write_manifest(out_path: Path, items: Iterable[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for obj in items:
            if not first:
                f.write(",\n")
            else:
                f.write("\n")
                first = False
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        f.write("\n]\n")


def _write_manifest_parallel(
    json_paths: Sequence[Path],
    out_path: Path,
    *,
    vocab: TagVocabulary,
    images_dir: Path,
    link_mode: str,
    workers: int,
) -> None:
    """Write a manifest by processing JSON sidecars in a thread pool and streaming results.

    This avoids holding the full list of entries in memory and accelerates IO-bound work
    (reading many small JSON files, creating symlinks) across shards.
    """

    def _process_one(jp: Path) -> Optional[dict]:
        try:
            # Use orjson if available (3-5x faster for batch processing)
            data = orjson.loads(jp.read_bytes()) if HAS_ORJSON else json.loads(jp.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            src, stem, ext = _find_image_for_json(jp, data)
            image_id, link_path = _make_unique_image_id(stem, ext, images_dir, src)
            try:
                _ensure_link(src, link_path, mode=link_mode)
            except FileExistsError:
                # Race with another worker; link now exists. Continue.
                pass
            except Exception:
                return None
            tags_list = parse_tags_field(data.get("tags"))
            labels = _labels_for_tags(tags_list, vocab)
            rating = data.get("rating", "unknown")
            return {"image_id": image_id, "labels": labels, "rating": rating}
        except Exception:
            return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("[")
        first = True
        # Use chunksize to amortize overhead on very large lists
        with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for i, obj in enumerate(ex.map(_process_one, json_paths, chunksize=1024), 1):
                if i % 10000 == 0:
                    logger.info("Processed %d / %d", i, len(json_paths))
                if not obj:
                    continue
                if not first:
                    f.write(",\n")
                else:
                    f.write("\n")
                    first = False
                f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        f.write("\n]\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate manifest files for OppaiOracle")
    ap.add_argument("--config", type=str, default="configs/unified_config.yaml", help="Path to unified config")
    ap.add_argument("--root", type=str, default=None, help="Dataset root; overrides config.data.storage_locations[*].path")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio (0..1) when not using cached split")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed; defaults to config.training.seed or 42")
    ap.add_argument("--images-dir", type=str, default=None, help="Images dir under root; defaults to <root>/images")
    ap.add_argument("--link-mode", type=str, choices=["symlink", "copy", "none"], default="symlink", help="How to materialize images/ entries")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N items (for smoke tests)")
    ap.add_argument("--no-split-cache", action="store_true", help="Ignore cached split lists; rescan and split")
    ap.add_argument("--workers", type=int, default=16, help="Thread workers for IO (JSON read/linking)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    root = Path(args.root) if args.root else _find_active_data_root(cfg)
    images_dir = Path(args.images_dir) if args.images_dir else (root / "images")

    seed = args.seed
    if seed is None:
        try:
            seed = int(getattr(getattr(cfg, "training"), "seed"))
        except Exception:
            seed = 42

    logger.info("Dataset root: %s", root)
    logger.info("Images dir: %s (mode=%s)", images_dir, args.link_mode)

    # Load vocabulary (JSON format with caching)
    vocab_path = Path(getattr(cfg, "vocab_path", "./vocabulary.json"))
    vocab = load_vocabulary_for_training(vocab_path)
    verify_vocabulary_integrity(vocab, vocab_path)
    logger.info("Loaded vocabulary with %d tags", len(vocab.tag_to_index))

    # Get split lists (cached or fresh)
    train_list, val_list = _iter_sidecar_jsons(root, use_cache=(not args.no_split_cache))
    if not val_list:
        # Fresh scan; split deterministically
        import random as _random
        rng = _random.Random(int(seed))
        rng.shuffle(train_list)
        split_ratio = float(args.val_ratio)
        n_val = max(1, int(len(train_list) * split_ratio))
        val_list = train_list[-n_val:]
        train_list = train_list[:-n_val]
        # Optionally persist split lists for reuse by loader
        train_file, val_file = _split_cache_paths(root)
        train_file.write_text("\n".join(str(p) for p in train_list), encoding="utf-8")
        val_file.write_text("\n".join(str(p) for p in val_list), encoding="utf-8")
        logger.info("Wrote split cache: train=%s, val=%s", train_file, val_file)

    if args.limit and args.limit > 0:
        train_list = train_list[: args.limit]
        val_list = val_list[: max(1, args.limit // 20)]
        logger.warning("Limit enabled: train=%d, val=%d", len(train_list), len(val_list))

    # Helper to yield manifest entries while creating links
    def build_entries(json_paths: Sequence[Path]):
        for i, jp in enumerate(json_paths, 1):
            if i % 10000 == 0:
                logger.info("Processed %d / %d", i, len(json_paths))
            try:
                # Use orjson if available (3-5x faster)
                data = orjson.loads(jp.read_bytes()) if HAS_ORJSON else json.loads(jp.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Skipping unreadable JSON %s: %s", jp, e)
                continue
            if not isinstance(data, dict):
                logger.warning("Skipping non-dict JSON %s", jp)
                continue
            # Resolve image source and destination
            try:
                src, stem, ext = _find_image_for_json(jp, data)
            except Exception as e:
                logger.warning("Skipping %s (image missing): %s", jp, e)
                continue
            # Create or reuse link
            image_id, link_path = _make_unique_image_id(stem, ext, images_dir, src)
            try:
                _ensure_link(src, link_path, mode=args.link_mode)
            except Exception as e:
                logger.warning("Failed to materialize image link for %s -> %s: %s", src, link_path, e)
                continue
            # Tags → indices
            tags_raw = data.get("tags")
            tags_list = parse_tags_field(tags_raw)
            labels = _labels_for_tags(tags_list, vocab)
            rating = data.get("rating", "unknown")
            yield {"image_id": image_id, "labels": labels, "rating": rating}

    # Stream write manifests
    train_out = root / "train.json"
    val_out = root / "val.json"
    logger.info("Writing manifests: %s, %s", train_out, val_out)
    # Prefer parallel writer for speed; fall back to single-thread generator if workers == 1
    if int(args.workers) > 1:
        _write_manifest_parallel(train_list, train_out, vocab=vocab, images_dir=images_dir, link_mode=args.link_mode, workers=int(args.workers))
        _write_manifest_parallel(val_list, val_out, vocab=vocab, images_dir=images_dir, link_mode=args.link_mode, workers=int(args.workers))
    else:
        _stream_write_manifest(train_out, build_entries(train_list))
        _stream_write_manifest(val_out, build_entries(val_list))
    logger.info("Done. Train entries: %d, Val entries: %d", len(train_list), len(val_list))


if __name__ == "__main__":
    main()
