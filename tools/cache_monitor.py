#!/usr/bin/env python3
"""
Standalone L2 cache monitor for OppaiOracle.

No wiring into the training/inference pipeline. This tool inspects the LMDB
environment used for L2, reports low‑overhead stats, and (optionally) estimates
coverage by counting (or sampling) keys that correspond to image entries versus
mask entries.

Usage examples:

  # Print basic LMDB stats
  python tools/cache_monitor.py --lmdb ./l2_cache --mode stats

  # Estimate coverage against an expected dataset size (skip filesystem scan)
  python tools/cache_monitor.py --lmdb ./l2_cache --mode coverage --expected 8000000

  # Watch fill rate and stats every 30s, write snapshots
  python tools/cache_monitor.py --lmdb ./l2_cache --mode watch --interval 30 --output ./cache_snapshots.jsonl

Notes:
  - Image entries use keys like: "<image_id>|cfg<hash>|flip<0/1>"
  - Mask entries append "|m". We don’t decode values; we only inspect keys.
  - For large DBs, use sampling (default) or pass --limit to bound key scans.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import lmdb  # type: ignore
except Exception:  # pragma: no cover
    lmdb = None


@dataclass
class LmdbStats:
    path: str
    map_size: int
    info_map_size: Optional[int]
    last_txnid: Optional[int]
    entries: int
    psize: int
    depth: int
    branch_pages: int
    leaf_pages: int
    overflow_pages: int


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:0.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}EB"


def open_env(path: str):
    if lmdb is None:
        print("LMDB is not installed. pip install lmdb", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(path):
        print(f"LMDB path not found or not a directory: {path}", file=sys.stderr)
        sys.exit(2)
    # Readonly open; no locks; allow multiple readers
    return lmdb.open(
        path,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        max_dbs=1,
    )


def get_stats(env) -> LmdbStats:
    st = env.stat()
    info = env.info()
    return LmdbStats(
        path=str(env.path()),
        map_size=int(env.map_size()),
        info_map_size=int(info.get("map_size", 0)) if isinstance(info, dict) else None,
        last_txnid=int(info.get("last_txnid", 0)) if isinstance(info, dict) else None,
        entries=int(st.get("entries", 0)),
        psize=int(st.get("psize", 0)),
        depth=int(st.get("depth", 0)),
        branch_pages=int(st.get("branch_pages", 0)),
        leaf_pages=int(st.get("leaf_pages", 0)),
        overflow_pages=int(st.get("overflow_pages", 0)),
    )


def estimate_image_entries(env, limit: int = 100000) -> tuple[int, int, float]:
    """Return (image_keys, mask_keys, mask_ratio_in_sample).

    Iterates up to `limit` keys and counts those ending with b"|m" as masks,
    others as image entries. For very large DBs, this provides a quick estimate.
    """
    image = 0
    mask = 0
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for i, (k, _v) in enumerate(cur.iternext(keys=True, values=False)):
            if i >= limit:
                break
            if k.endswith(b"|m"):
                mask += 1
            else:
                image += 1
    total = image + mask
    ratio = (mask / total) if total > 0 else 0.0
    return image, mask, ratio


def cmd_stats(args):
    env = open_env(args.lmdb)
    st = get_stats(env)
    print("LMDB Stats:")
    print(f"  path:              {st.path}")
    print(f"  entries:           {st.entries}")
    print(f"  map_size:          {human_bytes(st.map_size)}")
    if st.info_map_size:
        print(f"  info.map_size:     {human_bytes(st.info_map_size)}")
    if st.last_txnid is not None:
        print(f"  last_txnid:        {st.last_txnid}")
    print(f"  page size:         {st.psize} B")
    print(f"  depth:             {st.depth}")
    print(f"  leaf_pages:        {st.leaf_pages}")
    print(f"  branch_pages:      {st.branch_pages}")
    print(f"  overflow_pages:    {st.overflow_pages}")
    if args.sample or args.limit:
        img, m, r = estimate_image_entries(env, limit=int(args.limit or 100000))
        print("Sample (keys):")
        print(f"  images:            {img}")
        print(f"  masks:             {m}")
        print(f"  mask ratio:        {r:0.3f}")


def cmd_coverage(args):
    env = open_env(args.lmdb)
    st = get_stats(env)
    # Estimate split of entries via sampling (otherwise assume 50/50 image/mask)
    img, m, r = estimate_image_entries(env, limit=int(args.limit or 100000))
    if img + m == 0:
        # fallback heuristic
        est_images = st.entries // 2
    else:
        mask_ratio = (m / (img + m))
        est_images = int(st.entries * (1.0 - mask_ratio))
    expected = int(args.expected) if args.expected else None
    print("Coverage Estimate:")
    print(f"  total entries:     {st.entries}")
    print(f"  est. images:       {est_images}")
    if expected:
        cov = est_images / expected
        print(f"  expected images:   {expected}")
        print(f"  coverage:          {cov*100:0.2f}%")
    if args.output:
        snap = {
            "ts": time.time(),
            "path": st.path,
            "entries": st.entries,
            "est_images": est_images,
            "expected": expected,
            "map_size": st.map_size,
        }
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap) + "\n")


def cmd_watch(args):
    env = open_env(args.lmdb)
    prev = None
    interval = max(1.0, float(args.interval or 30.0))
    limit = int(args.limit or 50000)
    try:
        while True:
            st = get_stats(env)
            img, m, r = estimate_image_entries(env, limit=limit)
            if img + m == 0:
                est_images = st.entries // 2
            else:
                est_images = int(st.entries * (1.0 - (m / (img + m))))
            now = time.time()
            rate = None
            if prev is not None:
                dt = now - prev["ts"]
                d_entries = st.entries - prev["entries"]
                d_images = est_images - prev["est_images"]
                rate = {
                    "entries_per_s": d_entries / dt if dt > 0 else 0.0,
                    "images_per_s": d_images / dt if dt > 0 else 0.0,
                }
            print(f"[{time.strftime('%H:%M:%S')}] entries={st.entries} est_images={est_images} map={human_bytes(st.map_size)}")
            if rate:
                print(f"  + {rate['entries_per_s']:.1f} entries/s, {rate['images_per_s']:.1f} images/s (approx)")
            if args.output:
                snap = {
                    "ts": now,
                    "entries": st.entries,
                    "est_images": est_images,
                    "map_size": st.map_size,
                    "rate": rate,
                }
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snap) + "\n")
            prev = {"ts": now, "entries": st.entries, "est_images": est_images}
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped.")


def main():
    p = argparse.ArgumentParser(description="Standalone LMDB (L2) cache monitor")
    p.add_argument("--lmdb", required=True, help="Path to L2 LMDB directory")
    p.add_argument("--mode", choices=["stats", "coverage", "watch"], default="stats")
    p.add_argument("--expected", type=int, help="Expected number of images (for coverage calc)")
    p.add_argument("--limit", type=int, default=100000, help="Max keys to sample when estimating image vs mask")
    p.add_argument("--sample", action="store_true", help="In stats mode, also sample keys to show image/mask split")
    p.add_argument("--interval", type=float, default=30.0, help="Polling interval for watch mode (seconds)")
    p.add_argument("--output", help="Append JSON snapshots to this file")
    args = p.parse_args()

    if args.mode == "stats":
        cmd_stats(args)
    elif args.mode == "coverage":
        cmd_coverage(args)
    else:
        cmd_watch(args)


if __name__ == "__main__":
    main()

