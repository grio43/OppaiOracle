"""One-shot report: image_review correction counts, key buckets, and per-tag join with dataset frequencies."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS = ROOT / "image_review" / "corrections.json"
FREQ_FILE = ROOT / "logs" / "vocab_cache" / "2df7eb3c2bc0d784.frequencies.json"

NUMERIC_KEY = re.compile(r"^\d+$")
RUN_STEP_KEY = re.compile(r"^(?P<run>.+?)/step_(?P<step>\d+)/sample_(?P<idx>\d+)$")


def main() -> None:
    raw = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    corrections = raw.get("corrections", {}) if isinstance(raw, dict) else {}

    freq = json.loads(FREQ_FILE.read_text(encoding="utf-8"))

    n_entries = len(corrections)
    add_total = 0
    remove_total = 0
    add_per_tag: Counter[str] = Counter()
    remove_per_tag: Counter[str] = Counter()

    numeric_keys = 0
    run_step_keys: Counter[str] = Counter()
    other_keys = 0

    for key, entry in corrections.items():
        adds = entry.get("add", []) or []
        rems = entry.get("remove", []) or []
        add_total += len(adds)
        remove_total += len(rems)
        for t in adds:
            add_per_tag[t] += 1
        for t in rems:
            remove_per_tag[t] += 1

        if NUMERIC_KEY.match(key):
            numeric_keys += 1
            continue
        m = RUN_STEP_KEY.match(key)
        if m:
            run_step_keys[m.group("run")] += 1
        else:
            other_keys += 1

    print("=" * 70)
    print("IMAGE REVIEW CORRECTIONS REPORT")
    print("=" * 70)
    print(f"Source: {CORRECTIONS.relative_to(ROOT)}")
    print()
    print(f"Total reviewed images (entries): {n_entries:,}")
    print(f"  Total tag-add edits           : {add_total:,}")
    print(f"  Total tag-remove edits        : {remove_total:,}")
    print(f"  Total individual tag edits    : {add_total + remove_total:,}")
    print()

    print("Correction-key buckets")
    print("-" * 70)
    print(f"  numeric image_id keys (real dataset images) : {numeric_keys:,}")
    for run, n in run_step_keys.most_common():
        print(f"  run/step bucket  '{run}'  : {n:,}")
    if other_keys:
        print(f"  other / unrecognized keys                   : {other_keys:,}")
    print()

    all_tags = set(add_per_tag) | set(remove_per_tag)
    print(f"Unique tags touched by edits: {len(all_tags):,}")
    print(f"  (of which appear in dataset frequency map: "
          f"{sum(1 for t in all_tags if t in freq):,})")
    print()

    rows = []
    for tag in all_tags:
        adds = add_per_tag.get(tag, 0)
        rems = remove_per_tag.get(tag, 0)
        rows.append((tag, freq.get(tag, 0), adds, rems, adds + rems))

    rows.sort(key=lambda r: (-r[4], -r[1], r[0]))

    print("Per-tag edit counts vs dataset frequency (sorted by total edits desc)")
    print("-" * 70)
    print(f"{'tag':<40} {'dataset_freq':>14} {'+add':>6} {'-rem':>6} {'edits':>7}")
    print("-" * 70)
    for tag, f, a, r, tot in rows:
        print(f"{tag:<40} {f:>14,} {a:>6,} {r:>6,} {tot:>7,}")


if __name__ == "__main__":
    main()
