"""Migrate a MARM store (~/.marm/marm_memory.db) into agent-memory.

Two things this has to get right:

1. Re-embed everything. MARM's vectors are 512-dim jina-v2-small-en; ours are
   384-dim multilingual-e5-small-ko. The two are not comparable, so the old BLOBs
   are dropped rather than copied.

2. Deduplicate `memories` against `log_entries`. MARM's `marm_log_entry` writes
   the same text into BOTH tables, so a naive copy of 99 memories + 97 logs yields
   ~97 near-duplicates. Measured on the source store: 88 of 97 log entries already
   have their text inside a memory. `memories` is treated as the primary record and
   a log entry is only carried over when its text is not already represented.

The source database is opened read-only and never written to.
"""

import argparse
import html
import re
import sqlite3
import sys
import uuid
from pathlib import Path

from . import db, embedding, korean

DEFAULT_SOURCE = Path.home() / ".marm" / "marm_memory.db"

# Fixed namespace so ids are a pure function of (source, content): re-running the
# migration reproduces the same ids, which keeps eval fixtures and any external
# references valid across rebuilds.
_NAMESPACE = uuid.UUID("6f1a0c2e-8d3b-4f57-9a21-5c7e0b4d8e13")

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Undo MARM's HTML escaping.

    MARM's memory-write path escapes content but its log-write path does not, so
    92 of 99 rows in `memories` store `&quot;` where the user typed `"` while all
    97 rows in `log_entries` store it raw. Two consequences, both fixed here:
    the stored text is corrupted (and indexes a junk `quot` token), and the two
    tables' copies of the same entry no longer compare equal, which let
    near-duplicates past the deduplication below.
    """
    return html.unescape(text)


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _stable_id(source: str, content: str) -> str:
    return uuid.uuid5(_NAMESPACE, f"{source}|{_norm(content)}").hex


def _topic_from(content: str) -> str:
    """First line of the body.

    MARM's `log_entries.topic` is deliberately not used: it holds a coarse category
    ("general" in 84 of 97 rows), not a title. That was harmless in MARM, but here
    topic is BM25-weighted 3x and gets its own embedding, so a topic of "general"
    would actively degrade retrieval.
    """
    return content.splitlines()[0][:120].strip()


def _session_from(raw: str | None) -> str:
    """Recover the session name from MARM's `session_name` column.

    MARM's "Session: [name]" prefix parsing occasionally swallowed the whole entry:
    two `log_entries` rows carry 1,529 and 1,232 characters of body text as their
    session name. In both, the intended name is the first line
    ("aidlc-v2-workshop-review", "lotteon-unicorngym-seeding"), so taking that and
    capping the length recovers it instead of creating a session named after an
    entire memory.
    """
    name = (raw or "").splitlines()[0].strip() if raw else ""
    return name[:80] or "default"


def _fingerprint(text: str) -> str:
    """First 120 normalized characters — enough to identify a duplicate log entry
    whose stored copy differs only in trailing formatting."""
    return _norm(text)[:120]


def _insert(con, *, session, project, topic, content, created_at, source) -> bool:
    """Import one record. Returns False if it was already here.

    The id is a digest of source+content precisely so that running this twice is
    safe: MARM kept being written to after the first migration, and the only way to
    collect what arrived since is to re-read the whole source and skip what landed
    before. Without the skip, a second run dies on the first duplicate primary key
    and imports nothing.
    """
    memory_id = _stable_id(source, content)
    if con.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
        return False
    con.execute(
        "INSERT INTO memories(id, session, project, topic, content, created_at, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (memory_id, session, project, topic, content, created_at, source),
    )
    con.execute(
        "INSERT INTO memories_fts(topic, body, memory_id) VALUES (?, ?, ?)",
        (korean.index_text(topic), korean.index_text(content), memory_id),
    )
    chunks = embedding.passages(topic, content)
    vectors = embedding.encode_passages(chunks)
    con.executemany(
        "INSERT INTO memory_chunks(memory_id, idx, text, embedding) VALUES (?, ?, ?, ?)",
        [
            (memory_id, i, text, embedding.to_blob(vec))
            for i, (text, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    return True


def migrate(source: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"MARM database not found: {source}")

    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    con = db.connect()

    seen: set[str] = set()
    stats = {"memories": 0, "log_entries": 0, "skipped_duplicate_logs": 0,
             "already_present": 0}

    for row in src.execute(
        "SELECT session_name, project, content, timestamp, context_type "
        "FROM memories ORDER BY timestamp"
    ):
        content = _clean(row["content"] or "").strip()
        if not content:
            continue
        seen.add(_fingerprint(content))
        if not _insert(
            con,
            session=_session_from(row["session_name"]),
            project=row["project"],
            topic=_topic_from(content),
            content=content,
            created_at=row["timestamp"],
            source=f"marm:memory:{row['context_type'] or 'general'}",
        ):
            stats["already_present"] += 1
            continue
        stats["memories"] += 1
        if stats["memories"] % 20 == 0:
            con.commit()
            print(f"  memories: {stats['memories']}", file=sys.stderr, flush=True)

    for row in src.execute(
        "SELECT session_name, project, summary, full_entry, entry_date "
        "FROM log_entries ORDER BY entry_date"
    ):
        content = _clean(row["full_entry"] or row["summary"] or "").strip()
        if not content:
            continue
        fp = _fingerprint(content)
        if fp in seen:
            stats["skipped_duplicate_logs"] += 1
            continue
        seen.add(fp)
        if not _insert(
            con,
            session=_session_from(row["session_name"]),
            project=row["project"],
            topic=_topic_from(content),
            content=content,
            created_at=row["entry_date"],
            source="marm:log",
        ):
            stats["already_present"] += 1
            continue
        stats["log_entries"] += 1

    con.commit()
    stats["total"] = stats["memories"] + stats["log_entries"]
    stats["chunks"] = con.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
    src.close()
    con.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a MARM store into agent-memory")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    stats = migrate(args.source)
    print(f"migrated {stats['total']} memories "
          f"({stats['memories']} from memories, {stats['log_entries']} from log_entries)")
    print(f"skipped {stats['skipped_duplicate_logs']} log entries already present as memories")
    print(f"skipped {stats['already_present']} records already migrated")
    print(f"{stats['chunks']} chunks embedded")


if __name__ == "__main__":
    main()
