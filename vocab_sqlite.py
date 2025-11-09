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
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, Optional
from urllib.parse import quote

from vocabulary import TagVocabulary

logger = logging.getLogger(__name__)


def _connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Create a database connection with proper URI encoding."""
    db_path = Path(db_path)

    # Validate that path is absolute and normalized
    try:
        db_path = db_path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid database path: {db_path}") from e

    if read_only:
        # Properly encode the path component for URI
        # Convert to absolute POSIX path and URL-encode it
        posix_path = db_path.as_posix()
        # Ensure absolute path (required for file: URI)
        if not posix_path.startswith('/'):
            raise ValueError(f"Database path must be absolute for read-only mode: {db_path}")

        # URL-encode the path, but keep the forward slashes
        # quote() with safe='/' preserves path separators
        encoded_path = quote(posix_path, safe='/')

        # Construct URI with encoded path
        uri = f"file:{encoded_path}?mode=ro"
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


def save_vocabulary_to_sqlite(vocab: TagVocabulary, db_path: Path, *, update_freq: bool = True, strict: bool = False) -> None:
    """Persist TagVocabulary mapping into SQLite (append-safe).

    - Writes all (id, tag) pairs into tags table via INSERT OR REPLACE to
      keep ids stable.
    - Optionally writes frequencies into freq table.
    - Stores a compact JSON snapshot hash in meta for debugging.

    Args:
        vocab: The vocabulary to save
        db_path: Destination database path
        update_freq: Whether to update frequency counts
        strict: If True, raise on errors; if False, log warnings

    Raises:
        ValueError: If vocabulary is invalid
        sqlite3.Error: If database operation fails
    """
    # Validate vocabulary before attempting to save
    if not vocab.index_to_tag:
        raise ValueError("Cannot save empty vocabulary")

    ensure_schema(db_path)

    with _connect(db_path, read_only=False) as con:
        cur = con.cursor()

        try:
            # Write tags in deterministic id order
            tag_count = 0
            for idx in sorted(vocab.index_to_tag.keys()):
                tag = vocab.index_to_tag[idx]
                cur.execute(
                    "INSERT OR REPLACE INTO tags(id, tag) VALUES(?, ?)",
                    (int(idx), str(tag))
                )
                tag_count += 1

            logger.debug(f"Saved {tag_count} tags to database")

            # Update frequencies
            if update_freq:
                freq_updated = 0
                freq_skipped = 0

                for tag, count in (vocab.tag_frequencies or {}).items():
                    idx = vocab.tag_to_index.get(tag)
                    if idx is None:
                        freq_skipped += 1
                        logger.warning(
                            f"Tag '{tag}' has frequency count but no index mapping. "
                            f"This indicates vocabulary inconsistency."
                        )
                        if strict:
                            raise ValueError(
                                f"Vocabulary inconsistency: tag '{tag}' in frequencies "
                                f"but not in tag_to_index"
                            )
                        continue

                    cur.execute(
                        "INSERT OR REPLACE INTO freq(tag_id, count) VALUES(?, ?)",
                        (int(idx), int(count))
                    )
                    freq_updated += 1

                if freq_skipped > 0 or freq_updated > 0:
                    logger.debug(
                        f"Updated {freq_updated} frequency counts, skipped {freq_skipped}"
                    )

            # Store snapshot for traceability
            try:
                ordered = [
                    vocab.index_to_tag[i]
                    for i in sorted(vocab.index_to_tag.keys())
                ]

                if not ordered:
                    logger.warning("Empty vocabulary, cannot create snapshot")
                    snap = json.dumps({"count": 0, "pad": None})
                else:
                    snap = json.dumps({
                        "count": len(ordered),
                        "pad": ordered[0],
                        "unk": ordered[1] if len(ordered) > 1 else None
                    })

                cur.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("snapshot", snap)
                )
                logger.debug("Saved vocabulary snapshot to metadata")

            except (json.JSONEncodeError, KeyError, IndexError) as e:
                error_msg = f"Failed to create vocabulary snapshot: {e}"
                logger.error(error_msg)
                if strict:
                    raise ValueError(error_msg) from e

            # Commit all changes
            con.commit()
            logger.info(
                f"Successfully saved vocabulary to {db_path} "
                f"({tag_count} tags, update_freq={update_freq})"
            )

        except sqlite3.Error as e:
            # Rollback on database errors
            con.rollback()
            logger.error(f"Database error while saving vocabulary: {e}")
            raise


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

    Thread-safe and process-safe using database-level locking.

    Returns (new_tags_added, total_known_tags).

    Raises:
        sqlite3.DatabaseError: If database operations fail
    """
    ensure_schema(db_path)

    # Use IMMEDIATE transaction to acquire write lock immediately
    with _connect(db_path, read_only=False) as con:
        # Set transaction to IMMEDIATE to prevent concurrent writes
        con.execute("BEGIN IMMEDIATE")

        try:
            cur = con.cursor()

            # Fetch existing mapping with lock held
            cur.execute("SELECT id, tag FROM tags ORDER BY id")
            rows = cur.fetchall()
            existing = {str(tag): int(idx) for idx, tag in rows}

            # Calculate next ID safely
            if existing:
                max_id = max(existing.values())
                nxt = max_id + 1
            else:
                # First tags, reserve 0 and 1 for specials
                nxt = 2

            # Identify new tags deterministically
            new_tags = [t for t in tag_counts.keys() if t not in existing]
            new_tags.sort(key=lambda t: (-int(tag_counts.get(t, 0)), str(t)))

            # Insert new tags atomically
            new_count = 0
            for t in new_tags:
                try:
                    cur.execute(
                        "INSERT INTO tags(id, tag) VALUES(?, ?)",
                        (int(nxt), str(t))
                    )
                    existing[str(t)] = int(nxt)
                    nxt += 1
                    new_count += 1
                except sqlite3.IntegrityError as e:
                    # Should not happen with IMMEDIATE lock, but handle gracefully
                    logger.error(
                        f"Failed to insert tag '{t}' with id {nxt}: {e}. "
                        f"Possible concurrent modification."
                    )
                    # Re-read to get actual state
                    cur.execute("SELECT id FROM tags WHERE tag = ?", (str(t),))
                    result = cur.fetchone()
                    if result:
                        existing[str(t)] = int(result[0])
                    # Don't increment nxt, try next tag with same ID

            # Update frequencies for all known tags
            for t, idx in existing.items():
                c = int(tag_counts.get(t, 0))
                cur.execute(
                    "INSERT OR REPLACE INTO freq(tag_id, count) VALUES(?, ?)",
                    (int(idx), c)
                )

            # Commit transaction
            con.commit()

            logger.info(
                f"Appended {new_count} new tags, total {len(existing)} tags in vocabulary"
            )

            return (new_count, len(existing))

        except Exception as e:
            # Rollback on any error
            con.rollback()
            logger.error(f"Failed to append tags: {e}")
            raise


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
