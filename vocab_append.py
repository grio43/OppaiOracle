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
    """Find the first enabled storage location in configuration.

    Args:
        config: Configuration object with 'data' section

    Returns:
        Path to active data root

    Raises:
        RuntimeError: If no valid enabled storage location found
        ValueError: If configuration format is invalid
    """
    data_cfg = getattr(config, "data", None)
    if not data_cfg:
        raise RuntimeError(
            "Configuration has no 'data' section. "
            "Check that configs/unified_config.yaml is properly formatted."
        )

    storage_locations = getattr(data_cfg, "storage_locations", []) or []

    if not storage_locations:
        raise RuntimeError(
            "No storage locations defined in config.data.storage_locations. "
            "Add at least one storage location with 'enabled: true' and 'path: ...'."
        )

    valid_locations = []
    invalid_locations = []

    for i, loc in enumerate(storage_locations):
        try:
            # Parse location (support both dict and object formats)
            if isinstance(loc, dict):
                enabled = loc.get("enabled", False)
                path = loc.get("path")
            elif hasattr(loc, "enabled") and hasattr(loc, "path"):
                enabled = getattr(loc, "enabled", False)
                path = getattr(loc, "path", None)
            else:
                # Invalid format
                invalid_locations.append(
                    f"Location {i}: Invalid format (not dict or object with enabled/path)"
                )
                continue

            # Validate types
            if not isinstance(enabled, (bool, int)):
                invalid_locations.append(
                    f"Location {i}: 'enabled' must be boolean, got {type(enabled).__name__}"
                )
                continue

            if path is not None and not isinstance(path, (str, Path)):
                invalid_locations.append(
                    f"Location {i}: 'path' must be string or Path, got {type(path).__name__}"
                )
                continue

            # Convert to boolean
            enabled = bool(enabled)

            if enabled and path:
                valid_locations.append((i, Path(str(path))))
            elif enabled and not path:
                logger.warning(
                    f"Storage location {i} is enabled but has no path. Skipping."
                )

        except Exception as e:
            # Unexpected error - log full details for debugging
            logger.error(
                f"Unexpected error processing storage location {i}: {e}",
                exc_info=True
            )
            invalid_locations.append(
                f"Location {i}: Unexpected error: {e}"
            )
            continue

    # Report invalid locations
    if invalid_locations:
        error_msg = (
            "Invalid storage location configurations:\n" +
            "\n".join(f"  - {err}" for err in invalid_locations)
        )
        logger.error(error_msg)
        # Continue processing, maybe some locations are valid

    # Return first valid enabled location
    if valid_locations:
        index, path = valid_locations[0]
        logger.info(f"Using storage location {index}: {path}")
        return path

    # No enabled locations found
    if len(storage_locations) == len(invalid_locations):
        # All locations were invalid
        raise ValueError(
            f"All {len(storage_locations)} storage locations have invalid format. "
            f"See errors above."
        )
    else:
        # Locations exist but none are enabled
        raise RuntimeError(
            f"No enabled storage location found. "
            f"Found {len(storage_locations)} locations, {len(invalid_locations)} invalid. "
            f"Set 'enabled: true' for at least one storage location."
        )


def _scan_tags(root: Path) -> Counter:
    """Scan JSON files and count tags with proper error handling."""
    counts: Counter = Counter()

    if not root.exists():
        logger.warning(f"Dataset root does not exist: {root}")
        return counts

    json_files: List[Path] = sorted(root.rglob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {root}")
        return counts

    logger.info(f"Scanning {len(json_files)} JSON files in {root}...")

    # Track errors for reporting
    error_counts = {
        'json_errors': 0,
        'encoding_errors': 0,
        'permission_errors': 0,
        'other_errors': 0,
        'files_with_no_tags': 0,
        'successful': 0
    }

    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            file_had_tags = False

            # Sidecar (dict with 'tags')
            if isinstance(data, dict):
                tags_raw = data.get("tags")
                if tags_raw is not None:
                    tags = parse_tags_field(tags_raw)
                    if tags:
                        file_had_tags = True
                        for t in tags:
                            counts[t] += 1

            # Manifest-like (list of entries with 'tags')
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and ("tags" in entry):
                        tags = parse_tags_field(entry.get("tags"))
                        if tags:
                            file_had_tags = True
                            for t in tags:
                                counts[t] += 1

            if not file_had_tags:
                error_counts['files_with_no_tags'] += 1
                logger.debug(f"No tags found in {jp}")

            error_counts['successful'] += 1

        except json.JSONDecodeError as e:
            error_counts['json_errors'] += 1
            logger.debug(f"JSON decode error in {jp}: {e}")

        except UnicodeDecodeError as e:
            error_counts['encoding_errors'] += 1
            logger.warning(f"Encoding error in {jp}: {e}")

        except PermissionError as e:
            error_counts['permission_errors'] += 1
            logger.error(f"Permission denied reading {jp}: {e}")

        except Exception as e:
            error_counts['other_errors'] += 1
            logger.error(f"Unexpected error reading {jp}: {e}", exc_info=True)

    # Report summary
    total_errors = (
        error_counts['json_errors'] +
        error_counts['encoding_errors'] +
        error_counts['permission_errors'] +
        error_counts['other_errors']
    )

    if total_errors > 0:
        logger.warning(
            f"Scanned {len(json_files)} files, "
            f"{error_counts['successful']} successful, "
            f"{total_errors} errors "
            f"(JSON: {error_counts['json_errors']}, "
            f"encoding: {error_counts['encoding_errors']}, "
            f"permission: {error_counts['permission_errors']}, "
            f"other: {error_counts['other_errors']})"
        )

        # Warn if high error rate
        error_rate = total_errors / len(json_files)
        if error_rate > 0.1:  # More than 10% errors
            logger.error(
                f"High error rate: {error_rate*100:.1f}% of files failed to parse. "
                f"Check file permissions and JSON validity."
            )
    else:
        logger.info(
            f"Successfully scanned {error_counts['successful']} files, "
            f"found {len(counts)} unique tags"
        )

    if error_counts['files_with_no_tags'] > 0:
        logger.info(
            f"Note: {error_counts['files_with_no_tags']} files had no tags "
            f"(may be metadata-only files)"
        )

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
    logger.info(f"Scanning tags in {data_root}...")
    counts = _scan_tags(data_root)

    # First check: no tags found at all
    if not counts:
        raise RuntimeError(
            f"No tags found while scanning JSON sidecars under {data_root}. "
            f"Check that your dataset path is correct and contains JSON metadata."
        )

    original_count = len(counts)
    original_total = sum(counts.values())

    logger.info(
        f"Found {original_count} unique tags "
        f"({original_total} total occurrences) in dataset"
    )

    # Exclude ignored tags entirely
    ignored_count = 0
    ignored_occurrences = 0

    for ignored in list(vocab.ignored_tags):
        if ignored in counts:
            ignored_occurrences += counts[ignored]
            del counts[ignored]
            ignored_count += 1

    if ignored_count > 0:
        logger.info(
            f"Filtered out {ignored_count} ignored tags "
            f"({ignored_occurrences} occurrences)"
        )

    # Second check: no valid tags after filtering
    if not counts:
        raise RuntimeError(
            f"No valid tags remaining after filtering. "
            f"Dataset had {original_count} unique tags ({original_total} total occurrences), "
            f"but all were in the ignore list. "
            f"Check your ignore list configuration or dataset contents."
        )

    logger.info(
        f"Proceeding with {len(counts)} valid tags "
        f"({sum(counts.values())} total occurrences)"
    )

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
    logger.info(f"Saved vocabulary to {vocab_file}")

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
