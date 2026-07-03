"""One-shot restamp of `vocab_sha256` in existing checkpoints.

Older checkpoints stored `vocab_sha256` as a hash of the compact embedded
JSON bytes (model_metadata.py before the canonical-SHA fix). The on-disk
vocabulary.json hashes differently because of indentation / line endings,
so the resume guard refuses to load. After the fix, both ends compute the
SHA over the canonical form (sorted, compact, str-keyed JSON).

This script rewrites the `vocab_sha256` field of each checkpoint to the
canonical value derived from its already-embedded `vocab_b64_gzip` blob.
The model weights, optimizer state, and embedded vocab bytes are untouched.

Usage:
    python tools/restamp_vocab_sha.py path/to/ckpt.pt [more.pt ...]
    python tools/restamp_vocab_sha.py --dry-run path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from schemas import canonical_vocab_bytes  # noqa: E402


def restamp(ckpt_path: Path, dry_run: bool = False) -> bool:
    print(f"\n=== {ckpt_path} ===")
    if not ckpt_path.exists():
        print(f"  SKIP: file not found")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"  SKIP: checkpoint is not a dict ({type(ckpt).__name__})")
        return False
    if "vocab_b64_gzip" not in ckpt:
        print(f"  SKIP: no embedded vocab_b64_gzip")
        return False

    old_sha = ckpt.get("vocab_sha256", "<missing>")
    vocab_bytes = gzip.decompress(base64.b64decode(ckpt["vocab_b64_gzip"]))
    legacy_sha = hashlib.sha256(vocab_bytes).hexdigest()
    if legacy_sha != old_sha and old_sha != "<missing>":
        print(f"  WARN: stored vocab_sha256 ({old_sha}) does not match "
              f"sha256(embedded compact bytes) ({legacy_sha}). Embedded blob "
              f"may have been altered.")

    vocab_data = json.loads(vocab_bytes.decode("utf-8"))
    new_sha = hashlib.sha256(canonical_vocab_bytes(vocab_data)).hexdigest()

    print(f"  old vocab_sha256: {old_sha}")
    print(f"  new vocab_sha256: {new_sha}")
    print(f"  num_tags:         {ckpt.get('num_tags', len(vocab_data.get('tag_to_index', {})))}")

    if old_sha == new_sha:
        print(f"  already canonical, nothing to do")
        return True

    if dry_run:
        print(f"  DRY RUN: would update")
        return True

    sidecar = ckpt_path.with_suffix(ckpt_path.suffix + ".vocab_sha_backup.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(ckpt_path),
            "old_vocab_sha256": old_sha,
            "new_vocab_sha256": new_sha,
        }, f, indent=2)
    print(f"  wrote backup sidecar: {sidecar}")

    ckpt["vocab_sha256"] = new_sha
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)
    print(f"  rewrote checkpoint")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ok = True
    for p in args.checkpoints:
        ok &= restamp(p, dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
