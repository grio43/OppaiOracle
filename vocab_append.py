#!/usr/bin/env python3
"""
Append-only vocabulary updater for OppaiOracle.

Usage: python vocab_append.py

Behavior:
- Loads unified configuration from configs/unified_config.yaml.
- Finds the first enabled data.storage_locations[*].path as the dataset root.
- Scans all JSON sidecars under that root and collects tag frequencies.
- Loads the existing vocabulary mapping (tag_to_index / index_to_tag / tag_frequencies).
- Preserves all existing indices and appends any new tags at the end.
- Updates tag_frequencies for all tags based on current dataset JSONs.
- Writes the updated vocabulary back to config.vocab_path.

Notes:
- Special tokens are preserved: <PAD>=0, <UNK>=1.
- Tags listed in Tags_ignore.txt are excluded (no label signal, no frequency).
- L2 cache does not need clearing: it stores images only, not labels.
- New tags are appended in descending frequency (ties broken lexicographically) for determinism.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from Configuration_System import load_config
from vocabulary import TagVocabulary, verify_vocabulary_integrity
from utils.metadata_ingestion import parse_tags_field
from vocab_utils import compute_vocab_hash
from pathlib import Path as _PathAlias

logger = logging.getLogger("vocab_append")


def _resolve_vocab_file_path(vocab_path_str: str) -> Path:
    p = Path(vocab_path_str)
    if p.is_dir() or (not p.exists() and p.suffix == ""):  # directory-like
        return p / "vocabulary.json"
    return p


def _load_or_init_vocab(vocab_file: Path) -> TagVocabulary:
    v = TagVocabulary()
    if vocab_file.exists():
        v.load_vocabulary(vocab_file)
    else:
        # Initialise mapping with special tokens to ensure valid structure.
        v.tag_to_index = {v.pad_token: 0, v.unk_token: 1}
        v.index_to_tag = {0: v.pad_token, 1: v.unk_token}
        v.unk_index = 1
        v.tag_frequencies = {}
    return v


def _find_active_data_root(config) -> Path:
    data_cfg = getattr(config, "data", None)
    if not data_cfg:
        raise RuntimeError("Config has no 'data' section")
    storage_locations = getattr(data_cfg, "storage_locations", []) or []
    for loc in storage_locations:
        try:
            enabled = bool(loc.get("enabled", False)) if isinstance(loc, dict) else bool(getattr(loc, "enabled", False))
            path = loc.get("path") if isinstance(loc, dict) else getattr(loc, "path", None)
        except AttributeError:
            # Expected if object doesn't have 'enabled' or 'path'
            enabled, path = False, None
        except (TypeError, ValueError) as e:
            # Unexpected type or value errors
            logger.warning(f"Invalid storage location configuration: {e}")
            enabled, path = False, None
        if enabled and path:
            return Path(str(path))
    raise RuntimeError("No enabled storage location found in config.data.storage_locations")


def _scan_tags(root: Path) -> Counter:
    counts: Counter = Counter()
    json_files: List[Path] = sorted(root.rglob("*.json")) if root.exists() else []
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Sidecar (dict with 'tags')
        if isinstance(data, dict):
            tags_raw = data.get("tags")
            if tags_raw is not None:
                for t in parse_tags_field(tags_raw):
                    counts[t] += 1
            continue
        # Manifest-like (list of entries with 'tags')
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and ("tags" in entry):
                    for t in parse_tags_field(entry.get("tags")):
                        counts[t] += 1
    return counts


def _ordered_vocab_list(v: TagVocabulary) -> List[str]:
    # Return ordered list of tags by index (skips PAD/UNK for hash stability if desired)
    max_idx = max(v.index_to_tag.keys()) if v.index_to_tag else -1
    return [v.index_to_tag.get(i, v.unk_token) for i in range(0, max_idx + 1)]


def main() -> None:
    # Load unified config from the canonical path
    cfg = load_config("configs/unified_config.yaml")

    data_root = _find_active_data_root(cfg)
    vocab_file = _resolve_vocab_file_path(getattr(cfg, "vocab_path", "./vocabulary.json"))
    vocab_file.parent.mkdir(parents=True, exist_ok=True)

    vocab = _load_or_init_vocab(vocab_file)

    # Count tags from dataset
    counts = _scan_tags(data_root)
    if not counts:
        raise RuntimeError(f"No tags found while scanning JSON sidecars under {data_root}")

    # Exclude ignored tags entirely
    for ignored in list(vocab.ignored_tags):
        if ignored in counts:
            del counts[ignored]

    # Prepare append-only additions
    existing: Dict[str, int] = dict(vocab.tag_to_index)
    # Ensure special tokens are present
    if vocab.pad_token not in existing:
        existing[vocab.pad_token] = 0
        vocab.index_to_tag[0] = vocab.pad_token
    if vocab.unk_token not in existing:
        existing[vocab.unk_token] = 1
        vocab.index_to_tag[1] = vocab.unk_token

    # Determine next index
    next_idx = max(existing.values(), default=1) + 1

    # Identify new tags (not in existing mapping)
    new_tags = [t for t in counts.keys() if t not in existing]
    # Deterministic append order: frequency desc, then lexicographic
    new_tags.sort(key=lambda t: (-counts[t], t))

    # Append new tags
    for t in new_tags:
        existing[t] = next_idx
        vocab.index_to_tag[next_idx] = t
        next_idx += 1

    # Replace mapping on vocab
    vocab.tag_to_index = existing
    vocab.unk_index = vocab.tag_to_index.get(vocab.unk_token, 1)

    # Update frequencies for all known tags (excluding specials)
    freqs: Dict[str, int] = {}
    for tag in vocab.tag_to_index.keys():
        if tag in (vocab.pad_token, vocab.unk_token):
            continue
        freqs[tag] = int(counts.get(tag, 0))
    vocab.tag_frequencies = freqs

    # Validate integrity (reject placeholder tags)
    verify_vocabulary_integrity(vocab, vocab_file)

    # Save vocabulary
    vocab.save_vocabulary(vocab_file)

    # Also mirror to SQLite (same directory, file name 'vocab.sqlite')
    try:
        from vocab_sqlite import save_vocabulary_to_sqlite
        sqlite_path = _PathAlias(vocab_file).with_name('vocab.sqlite')
        save_vocabulary_to_sqlite(vocab, sqlite_path)
        logger.info(f"Mirrored vocabulary to SQLite at {sqlite_path}")
    except Exception as e:
        logger.warning(f"Could not mirror vocabulary to SQLite: {e}")

    # Report summary
    ordered = _ordered_vocab_list(vocab)
    sha = compute_vocab_hash(ordered)
    print("✓ Vocabulary updated")
    print(f"Dataset root: {data_root}")
    print(f"Vocabulary file: {vocab_file}")
    print(f"Scanned JSON files: ~{len(list(data_root.rglob('*.json')))} (recursively)")
    print(f"Existing tags: {len(ordered)} (incl. specials)")
    print(f"New tags appended: {len(new_tags)}")
    if new_tags:
        preview: List[Tuple[str, int]] = [(t, counts.get(t, 0)) for t in new_tags[:25]]
        print("Top new tags (count):", ", ".join(f"{t}({c})" for t, c in preview))
    print(f"Vocab SHA256 (ordered list): {sha}")
    print("Note: L2 cache need not be cleared; it stores images only.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
