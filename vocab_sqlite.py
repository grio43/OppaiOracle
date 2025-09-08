#!/usr/bin/env python3
"""
SQLite backend for vocabulary storage.

Purpose:
- Provide a durable, append-only store for tag indices and frequencies.
- Preserve stable indices across runs to avoid training mismatches.
- Remain compatible with existing JSON embedding in checkpoints by
  supporting round-trip: TagVocabulary <-> SQLite <-> JSON.

Schema (single-file DB):
- meta(key TEXT PRIMARY KEY, value TEXT)
- tags(id INTEGER PRIMARY KEY, tag TEXT UNIQUE NOT NULL)
- freq(tag_id INTEGER PRIMARY KEY REFERENCES tags(id), count INTEGER DEFAULT 0)

Notes:
- Indices (id) match TagVocabulary indices exactly, including specials
  <PAD>=0 and <UNK>=1. Never renumber existing ids; only append new tags.
- This module is import-safe (no side effects at import time).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, Optional

from vocabulary import TagVocabulary


def _connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(db_path)
    if read_only:
        # URI mode for read-only connection
        uri = f"file:{db_path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)
    return sqlite3.connect(str(db_path), check_same_thread=False)


def ensure_schema(db_path: Path) -> None:
    """Create tables if they do not exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path, read_only=False) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
              id   INTEGER PRIMARY KEY,
              tag  TEXT UNIQUE NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS freq (
              tag_id INTEGER PRIMARY KEY,
              count  INTEGER DEFAULT 0,
              FOREIGN KEY(tag_id) REFERENCES tags(id)
            )
            """
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
        con.commit()


def save_vocabulary_to_sqlite(vocab: TagVocabulary, db_path: Path, *, update_freq: bool = True) -> None:
    """Persist TagVocabulary mapping into SQLite (append-safe).

    - Writes all (id, tag) pairs into tags table via INSERT OR REPLACE to
      keep ids stable.
    - Optionally writes frequencies into freq table.
    - Stores a compact JSON snapshot hash in meta for debugging.
    """
    ensure_schema(db_path)
    with _connect(db_path, read_only=False) as con:
        cur = con.cursor()
        # Write tags in deterministic id order
        for idx in sorted(vocab.index_to_tag.keys()):
            tag = vocab.index_to_tag[idx]
            cur.execute("INSERT OR REPLACE INTO tags(id, tag) VALUES(?, ?)", (int(idx), str(tag)))
        if update_freq:
            for tag, count in (vocab.tag_frequencies or {}).items():
                idx = vocab.tag_to_index.get(tag)
                if idx is None:
                    continue
                cur.execute("INSERT OR REPLACE INTO freq(tag_id, count) VALUES(?, ?)", (int(idx), int(count)))

        # Store a minimal snapshot for traceability
        try:
            ordered = [vocab.index_to_tag[i] for i in sorted(vocab.index_to_tag.keys())]
            snap = json.dumps({"count": len(ordered), "pad": ordered[0] if ordered else "<PAD>"})
            cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("snapshot", snap))
        except Exception:
            pass
        con.commit()


def load_vocabulary_from_sqlite(db_path: Path) -> TagVocabulary:
    """Load TagVocabulary from SQLite file.

    Returns a TagVocabulary with stable indices and any stored frequencies.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite vocabulary not found: {db_path}")

    with _connect(db_path, read_only=True) as con:
        cur = con.cursor()
        # Read mapping ordered by id to preserve index positions
        cur.execute("SELECT id, tag FROM tags ORDER BY id ASC")
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"Empty SQLite vocabulary: {db_path}")

        v = TagVocabulary()
        v.tag_to_index = {}
        v.index_to_tag = {}
        for idx, tag in rows:
            v.tag_to_index[str(tag)] = int(idx)
            v.index_to_tag[int(idx)] = str(tag)

        # Ensure specials exist (append minimally if missing)
        if v.pad_token not in v.tag_to_index:
            v.tag_to_index[v.pad_token] = 0
            v.index_to_tag[0] = v.pad_token
        if v.unk_token not in v.tag_to_index:
            v.tag_to_index[v.unk_token] = 1
            v.index_to_tag[1] = v.unk_token
        v.unk_index = v.tag_to_index.get(v.unk_token, 1)
        v.tags = [t for t in v.tag_to_index.keys() if t not in (v.pad_token, v.unk_token)]

        # Frequencies (optional)
        try:
            cur.execute("SELECT t.tag, f.count FROM freq f JOIN tags t ON t.id=f.tag_id")
            freq_rows = cur.fetchall()
            v.tag_frequencies = {str(tag): int(count) for tag, count in freq_rows}
        except Exception:
            v.tag_frequencies = {}

        return v


def next_index(db_path: Path) -> int:
    """Return next available tag id (max(id)+1 or 2 if only specials)."""
    with _connect(db_path, read_only=True) as con:
        cur = con.cursor()
        cur.execute("SELECT COALESCE(MAX(id), 1) FROM tags")
        (mx,) = cur.fetchone()
        return int(mx) + 1


def append_tags(db_path: Path, tag_counts: Dict[str, int]) -> Tuple[int, int]:
    """Append new tags with stable indices; update frequencies for all.

    Returns (new_tags_added, total_known_tags).
    """
    ensure_schema(db_path)
    with _connect(db_path, read_only=False) as con:
        cur = con.cursor()
        # Fetch existing mapping into memory
        cur.execute("SELECT id, tag FROM tags")
        existing = {str(tag): int(idx) for idx, tag in cur.fetchall()}

        # Ensure specials exist
        if "<PAD>" not in existing.values():
            pass  # Already handled by TagVocabulary side; keep minimal here

        nxt = (max(existing.values()) + 1) if existing else 2
        # Sort incoming tags deterministically: count desc, tag asc
        new_tags = [t for t in tag_counts.keys() if t not in existing]
        new_tags.sort(key=lambda t: (-int(tag_counts.get(t, 0)), str(t)))
        for t in new_tags:
            cur.execute("INSERT OR IGNORE INTO tags(id, tag) VALUES(?, ?)", (int(nxt), str(t)))
            existing[str(t)] = int(nxt)
            nxt += 1
        # Update freq for all known tags we have counts for
        for t, idx in existing.items():
            c = int(tag_counts.get(t, 0))
            cur.execute("INSERT OR REPLACE INTO freq(tag_id, count) VALUES(?, ?)", (int(idx), c))
        con.commit()
        return (len(new_tags), len(existing))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SQLite vocabulary utilities")
    sub = p.add_subparsers(dest="cmd")

    to_sql = sub.add_parser("from-json", help="Create/update SQLite from a vocabulary.json")
    to_sql.add_argument("json_path", type=str, help="Path to vocabulary.json")
    to_sql.add_argument("sqlite_path", type=str, nargs="?", default="vocab.sqlite", help="Destination SQLite path")

    dump = sub.add_parser("to-json", help="Export SQLite vocabulary to JSON")
    dump.add_argument("sqlite_path", type=str, help="Path to vocab.sqlite")
    dump.add_argument("json_path", type=str, nargs="?", default="vocabulary.json", help="Destination JSON path")

    args = p.parse_args()
    if args.cmd == "from-json":
        src = Path(args.json_path)
        dst = Path(args.sqlite_path)
        v = TagVocabulary()
        v.load_vocabulary(src)
        ensure_schema(dst)
        save_vocabulary_to_sqlite(v, dst)
        print(f"✓ Wrote SQLite vocabulary: {dst}")
    elif args.cmd == "to-json":
        src = Path(args.sqlite_path)
        dst = Path(args.json_path)
        v = load_vocabulary_from_sqlite(src)
        v.save_vocabulary(Path(dst))
        print(f"✓ Wrote JSON vocabulary: {dst}")
    else:
        p.print_help()
