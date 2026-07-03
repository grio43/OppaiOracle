"""Render image_review/corrections.json into a full markdown breakdown."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS = ROOT / "image_review" / "corrections.json"
FREQ_FILE = ROOT / "logs" / "vocab_cache" / "2df7eb3c2bc0d784.frequencies.json"
OUT = ROOT / "notes" / "corrections_report.md"

NUMERIC_KEY = re.compile(r"^\d+$")
RUN_STEP_KEY = re.compile(r"^(?P<run>.+?)/step_(?P<step>\d+)/sample_(?P<idx>\d+)$")

BANDS = [
    ("≥ 1M",          1_000_000, None),
    ("500k – 1M",       500_000, 1_000_000),
    ("100k – 500k",     100_000, 500_000),
    ("50k – 100k",       50_000, 100_000),
    ("10k – 50k",        10_000, 50_000),
    ("1k – 10k",          1_000, 10_000),
    ("1 – 1k",                1, 1_000),
    ("0 (not in freq map)",        0, 1),
]


def band_for(freq: int) -> str:
    for label, lo, hi in BANDS:
        if hi is None and freq >= lo:
            return label
        if hi is not None and lo <= freq < hi:
            return label
    return BANDS[-1][0]


def fmt(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    raw = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    corrections = raw.get("corrections", {}) if isinstance(raw, dict) else {}
    schema_version = raw.get("version") if isinstance(raw, dict) else None
    freq = json.loads(FREQ_FILE.read_text(encoding="utf-8"))

    n_entries = len(corrections)
    add_total = remove_total = 0
    add_per_tag: Counter[str] = Counter()
    remove_per_tag: Counter[str] = Counter()
    edits_per_image: list[tuple[str, int, int, str]] = []
    timestamps: list[str] = []

    numeric_keys = 0
    run_step_groups: dict[str, Counter[int]] = defaultdict(Counter)
    other_keys: list[str] = []

    edits_only_remove = 0
    edits_only_add = 0
    edits_both = 0
    edits_empty = 0

    for key, entry in corrections.items():
        adds = list(entry.get("add", []) or [])
        rems = list(entry.get("remove", []) or [])
        ts = entry.get("updated_at")
        if ts:
            timestamps.append(ts)
        add_total += len(adds)
        remove_total += len(rems)
        for t in adds:
            add_per_tag[t] += 1
        for t in rems:
            remove_per_tag[t] += 1
        edits_per_image.append((key, len(adds), len(rems), ts or ""))

        if adds and rems:
            edits_both += 1
        elif adds:
            edits_only_add += 1
        elif rems:
            edits_only_remove += 1
        else:
            edits_empty += 1

        if NUMERIC_KEY.match(key):
            numeric_keys += 1
            continue
        m = RUN_STEP_KEY.match(key)
        if m:
            run_step_groups[m.group("run")][int(m.group("step"))] += 1
        else:
            other_keys.append(key)

    all_tags = sorted(set(add_per_tag) | set(remove_per_tag))
    rows = []
    for tag in all_tags:
        a = add_per_tag.get(tag, 0)
        r = remove_per_tag.get(tag, 0)
        f = int(freq.get(tag, 0))
        rows.append((tag, f, a, r, a + r, band_for(f)))

    band_counts: Counter[str] = Counter()
    band_edits: Counter[str] = Counter()
    band_rows: dict[str, list] = defaultdict(list)
    for row in rows:
        tag, f, a, r, tot, band = row
        band_counts[band] += 1
        band_edits[band] += tot
        band_rows[band].append(row)

    rows.sort(key=lambda r: (-r[1], -r[4], r[0]))
    edits_per_image.sort(key=lambda r: (-(r[1] + r[2]), r[0]))

    ts_min = min(timestamps) if timestamps else None
    ts_max = max(timestamps) if timestamps else None
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out: list[str] = []
    p = out.append

    p("# Image Review Corrections — Full Breakdown")
    p("")
    p(f"_Generated {generated} from `{CORRECTIONS.relative_to(ROOT).as_posix()}` "
      f"(schema v{schema_version}) joined with `{FREQ_FILE.relative_to(ROOT).as_posix()}`._")
    p("")
    p("## 1. Headline numbers")
    p("")
    p(f"- **Reviewed images (correction entries):** {fmt(n_entries)}")
    p(f"- **Total tag-add edits:** {fmt(add_total)}")
    p(f"- **Total tag-remove edits:** {fmt(remove_total)}")
    p(f"- **Total individual tag edits:** {fmt(add_total + remove_total)}")
    p(f"- **Unique tags touched:** {fmt(len(all_tags))} "
      f"({fmt(sum(1 for t in all_tags if t in freq))} present in the dataset frequency map, "
      f"{fmt(sum(1 for t in all_tags if t not in freq))} absent)")
    if ts_min and ts_max:
        p(f"- **Edit timestamp range:** {ts_min} → {ts_max}")
    p("")
    p("### Per-image edit shape")
    p("")
    p("| Pattern | Image count |")
    p("|---|---:|")
    p(f"| Adds only | {fmt(edits_only_add)} |")
    p(f"| Removes only | {fmt(edits_only_remove)} |")
    p(f"| Both adds and removes | {fmt(edits_both)} |")
    p(f"| Empty (would have been auto-deleted) | {fmt(edits_empty)} |")
    p("")
    avg_per_image = (add_total + remove_total) / n_entries if n_entries else 0.0
    max_per_image = max((a + r for _, a, r, _ in edits_per_image), default=0)
    p(f"- **Mean edits per reviewed image:** {avg_per_image:.2f}")
    p(f"- **Max edits on a single image:** {max_per_image}")
    p("")

    p("## 2. Correction-key buckets")
    p("")
    p("Keys are either a numeric dataset `image_id` or the fallback "
      "`<run>/step_<step>/sample_<idx>` used when the run did not log an `image_id`.")
    p("")
    p("| Bucket | Entries |")
    p("|---|---:|")
    p(f"| Numeric `image_id` (dataset images) | {fmt(numeric_keys)} |")
    for run, steps in sorted(run_step_groups.items()):
        p(f"| `{run}/step_*/sample_*` | {fmt(sum(steps.values()))} |")
    if other_keys:
        p(f"| Other / unrecognized | {fmt(len(other_keys))} |")
    p("")
    if run_step_groups:
        p("### Fallback-keyed entries by step")
        p("")
        p("| Run | Step | Entries |")
        p("|---|---:|---:|")
        for run, steps in sorted(run_step_groups.items()):
            for step, n in sorted(steps.items()):
                p(f"| `{run}` | {step} | {fmt(n)} |")
        p("")
    if other_keys:
        p("### Unrecognized keys")
        p("")
        for k in sorted(other_keys):
            p(f"- `{k}`")
        p("")

    p("## 3. Edit distribution by dataset frequency")
    p("")
    p("| Frequency band | Unique tags | Total edits | Edits / tag |")
    p("|---|---:|---:|---:|")
    band_total_tags = band_total_edits = 0
    for label, _, _ in BANDS:
        n = band_counts.get(label, 0)
        e = band_edits.get(label, 0)
        per = (e / n) if n else 0.0
        band_total_tags += n
        band_total_edits += e
        p(f"| {label} | {fmt(n)} | {fmt(e)} | {per:.2f} |")
    p(f"| **Total** | **{fmt(band_total_tags)}** | **{fmt(band_total_edits)}** | "
      f"**{(band_total_edits / band_total_tags) if band_total_tags else 0:.2f}** |")
    p("")

    p("## 4. Add vs remove breakdown")
    p("")
    p("### 4.1 Tags that were *removed* from ground truth")
    p("")
    if remove_per_tag:
        p("| Tag | Dataset freq | Removes |")
        p("|---|---:|---:|")
        for tag, n in sorted(remove_per_tag.items(), key=lambda kv: (-kv[1], kv[0])):
            p(f"| `{tag}` | {fmt(int(freq.get(tag, 0)))} | {fmt(n)} |")
    else:
        p("_(none)_")
    p("")
    p("### 4.2 Tags that were *added* to ground truth")
    p("")
    p(f"_See section 5 for the full sorted list. Of {fmt(len(add_per_tag))} unique added tags, "
      f"{fmt(sum(1 for v in add_per_tag.values() if v == 1))} were added on exactly one image and "
      f"{fmt(sum(1 for v in add_per_tag.values() if v >= 10))} were added on 10+ images._")
    p("")

    p("## 5. Full per-tag table (sorted by dataset frequency desc, then total edits desc)")
    p("")
    p("| # | Tag | Dataset freq | Adds | Removes | Total edits | Freq band |")
    p("|---:|---|---:|---:|---:|---:|---|")
    for i, (tag, f, a, r, tot, band) in enumerate(rows, 1):
        p(f"| {i} | `{tag}` | {fmt(f)} | {fmt(a)} | {fmt(r)} | {fmt(tot)} | {band} |")
    p("")

    p("## 6. Per-band tag listings (sorted by dataset frequency desc within band)")
    p("")
    for label, _, _ in BANDS:
        rs = band_rows.get(label, [])
        if not rs:
            continue
        p(f"### {label} — {fmt(len(rs))} tags, {fmt(sum(x[4] for x in rs))} edits")
        p("")
        p("| Tag | Dataset freq | Adds | Removes | Total edits |")
        p("|---|---:|---:|---:|---:|")
        for tag, f, a, r, tot, _band in sorted(rs, key=lambda x: (-x[1], -x[4], x[0])):
            p(f"| `{tag}` | {fmt(f)} | {fmt(a)} | {fmt(r)} | {fmt(tot)} |")
        p("")

    p("## 7. Most-edited individual images")
    p("")
    p("Top 30 reviewed images by total edits (adds + removes).")
    p("")
    p("| # | Correction key | Adds | Removes | Total | Updated |")
    p("|---:|---|---:|---:|---:|---|")
    for i, (key, a, r, ts) in enumerate(edits_per_image[:30], 1):
        p(f"| {i} | `{key}` | {fmt(a)} | {fmt(r)} | {fmt(a + r)} | {ts} |")
    p("")

    p("## 8. Source files")
    p("")
    p(f"- Corrections store: `{CORRECTIONS.relative_to(ROOT).as_posix()}`")
    p(f"- Dataset tag frequencies: `{FREQ_FILE.relative_to(ROOT).as_posix()}`")
    p(f"- Generator script: `{Path(__file__).relative_to(ROOT).as_posix()}`")
    p("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}  ({len(out)} lines, {OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
