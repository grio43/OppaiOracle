#!/usr/bin/env python3
"""
Lightweight vocabulary builder for sidecar JSON datasets.

Scans a directory for .json files with a "tags" field and writes
vocabulary.json in the OppaiOracle repo root without importing PyTorch.

This mirrors the structure used by TagVocabulary.save_vocabulary:
- tag_to_index: dict[str,int] with <PAD>=0 and <UNK>=1
- index_to_tag: dict[int,str]
- tag_frequencies: dict[str,int]

Usage:
  python tools/generate_vocab_from_sidecars.py \
      --data /path/to/shard \
      --out /media/andrewk/qnap-public/workspace/OppaiOracle/vocabulary.json \
      --min-frequency 5 --top-k 100000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Iterable


def load_ignore_tags(ignore_file: Path) -> set[str]:
    ignored: set[str] = set()
    try:
        if ignore_file.exists():
            for line in ignore_file.read_text(encoding="utf-8").splitlines():
                tag = line.strip()
                if not tag or tag.startswith("#"):
                    continue
                ignored.add(tag)
    except Exception:
        pass
    return ignored


def iter_sidecar_jsons(root: Path) -> Iterable[Path]:
    # Scan only the top-level for speed, matching DatasetLoader sidecar mode.
    for p in root.iterdir():
        if p.suffix.lower() == ".json":
            yield p


def parse_tags(payload) -> list[str]:
    tags_field = payload.get("tags")
    if not tags_field:
        return []
    if isinstance(tags_field, str):
        return [t for t in tags_field.split() if t]
    if isinstance(tags_field, list):
        return [str(t) for t in tags_field if str(t)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Root folder containing sidecar JSON files")
    ap.add_argument("--out", type=Path, required=True, help="Path to write vocabulary.json")
    ap.add_argument("--min-frequency", type=int, default=5, help="Minimum occurrences to keep a tag")
    ap.add_argument("--top-k", type=int, default=100000, help="Cap number of tags (excl. special tokens)")
    ap.add_argument("--ignore-file", type=Path, default=Path(__file__).resolve().parent.parent / "Tags_ignore.txt",
                    help="Path to Tags_ignore.txt; defaults to repo root if present")
    args = ap.parse_args()

    root = args.data
    if not root.exists():
        raise SystemExit(f"Data path does not exist: {root}")

    ignored = load_ignore_tags(args.ignore_file)

    counts: Counter[str] = Counter()
    json_files = list(iter_sidecar_jsons(root))
    for jpath in json_files:
        try:
            payload = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for t in parse_tags(payload):
            if t in ignored:
                continue
            counts[t] += 1

    # Order by frequency desc, then lexicographically for determinism
    sorted_tags = sorted(
        (t for t, c in counts.items() if c >= args.min_frequency),
        key=lambda x: (-counts[x], x)
    )
    if args.top_k and args.top_k > 0:
        sorted_tags = sorted_tags[: args.top_k]

    tag_to_index: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    index_to_tag: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    for i, tag in enumerate(sorted_tags, start=2):
        tag_to_index[tag] = i
        index_to_tag[i] = tag

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "tag_to_index": tag_to_index,
            # JSON object keys must be strings; writing ints is fine, they load as strings.
            "index_to_tag": {int(k): v for k, v in index_to_tag.items()},
            "tag_frequencies": {t: counts[t] for t in sorted_tags},
        }, f, ensure_ascii=False, indent=2)

    print(f"Wrote vocabulary with {len(tag_to_index)} entries to {out_path}")


if __name__ == "__main__":
    main()

