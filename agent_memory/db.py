"""SQLite schema and connection handling."""

import sqlite3

from . import config

SCHEMA_VERSION = 7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    session     TEXT NOT NULL,
    project     TEXT,
    topic       TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'write',
    -- 'cold' = archived in this database, 'hot' = mirrored from a Claude Code
    -- memory file, which is the source of truth. See hot.py.
    tier        TEXT NOT NULL DEFAULT 'cold',
    path        TEXT,
    -- What replaced this memory, when it has been corrected by a later one. A
    -- reference, not a foreign key: a hot section's id is a hash of its body and dies
    -- on the next edit of that file, so `path#heading` has to be expressible too. See
    -- store.supersede.
    --
    -- Ranking ignores this on purpose. A flat penalty is the mirror image of the hot
    -- boost that was measured and rejected, and the fixture case this exists for is a
    -- cold memory that *deserves* its rank -- it is the fuller account, it is just no
    -- longer current. What the reader needs is to be told, not to have it hidden.
    superseded_by TEXT,
    superseded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);

-- Change detection for the hot mirror, so a recall re-embeds only edited files.
--
-- `project` is here because several projects share this store and a memory's project
-- was taken from the *current working directory*. That is right for the usual case --
-- each server process mirrors its own project -- and wrong the moment a process
-- indexes a directory belonging to someone else, which AGENT_MEMORY_HOT_PATHS exists
-- to allow. Measured: pointing it at another project's memory dir relabelled that
-- project's section as this one's, which silently breaks `project=` filtering. The
-- directory's first indexer records the name here and later ones inherit it.
CREATE TABLE IF NOT EXISTS hot_files (
    path    TEXT PRIMARY KEY,
    mtime   REAL NOT NULL,
    size    INTEGER NOT NULL,
    hash    TEXT NOT NULL,
    project TEXT
);

-- Standalone (not external-content) because the indexed text is derived by
-- korean.index_text(), not a column of `memories`. FTS5 cannot compute it, so
-- writes maintain this table explicitly.
--
-- The link is an explicit UNINDEXED memory_id, not rowid parity. Parity is an
-- invariant nothing enforces: deleting a memory without deleting its FTS row
-- leaves an orphan, SQLite later reuses that rowid for a different memory, and the
-- lexical lane silently returns the wrong record. That happened -- hot-file
-- re-indexing hit it, and lexical recall dropped to R@5 0% before the crash
-- surfaced it. An explicit key cannot drift.
--
-- topic and body are separate columns so bm25() can weight them differently. The
-- topic line is a keyword-dense summary of what a memory is *about*, which is what
-- paraphrase queries ask for; measured on 20 hand-authored paraphrase queries,
-- weighting it 3x lifted lexical R@1 from 30% to 40%.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(topic, body, memory_id UNINDEXED, tokenize='unicode61');

CREATE TABLE IF NOT EXISTS memory_chunks (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    PRIMARY KEY (memory_id, idx)
);

CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- Concept graph. Entities and relations are supplied by the writing agent, not
-- extracted from prose here. MARM extracted with spaCy and it did not survive
-- contact with Korean: measured on this corpus, marm_concept_build produced 361
-- entities and 0 relations, with boundaries cut mid-particle ("결론은 marm-memory",
-- "캡처)는") because en_core_web_sm cannot split 은/는/가/를. spaCy's Korean model
-- is not a fix either -- it has no noun_chunks at all (syntax_iterators == {}),
-- which is where that pipeline starts.
--
-- `name` is the normalized join key; `display` keeps the surface form for output.
CREATE TABLE IF NOT EXISTS entities (
    name        TEXT PRIMARY KEY,
    display     TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'concept',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity      TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity);

-- memory_id is the evidence: which memory asserted this relation. Keeping it in the
-- key means the same relation stated by two memories is two rows, so a relation can
-- be traced back to what said it and disappears when that memory does.
CREATE TABLE IF NOT EXISTS relations (
    source      TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
    predicate   TEXT NOT NULL,
    target      TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (source, predicate, target, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target);
"""


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(_SCHEMA)
    _migrate(con)
    _init_meta(con)
    con.commit()
    return con


def _migrate(con: sqlite3.Connection) -> None:
    """Bring an existing file up to the current schema.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so a
    column added to the DDL above never appears in a store created before it.
    """
    columns = {row[1] for row in con.execute("PRAGMA table_info(memories)")}
    for column in ("superseded_by", "superseded_at"):
        if column not in columns:
            con.execute(f"ALTER TABLE memories ADD COLUMN {column} TEXT")

    columns = {row[1] for row in con.execute("PRAGMA table_info(hot_files)")}
    if "project" not in columns:
        con.execute("ALTER TABLE hot_files ADD COLUMN project TEXT")
        # Backfill from the rows that file produced. Those already carry the right
        # project for every directory each project indexed itself.
        con.execute(
            "UPDATE hot_files SET project = ("
            "  SELECT m.project FROM memories m"
            "  WHERE m.path = hot_files.path AND m.project IS NOT NULL LIMIT 1)"
        )


def _init_meta(con: sqlite3.Connection) -> None:
    """Record the encoder the stored vectors were produced by.

    Embeddings are only comparable within one model+dimension. Persisting this is
    what lets `mem_recall` refuse to silently rank against stale vectors after a
    model swap -- MARM had no such record, so its 512-dim jina vectors and any
    replacement would have been compared as if interchangeable.
    """
    # embed_model / embed_dim describe the vectors already on disk, so they are
    # written once and never overwritten — that is exactly what lets
    # embedding_mismatch() notice a model swap instead of ranking against stale
    # vectors. schema_version describes the tables, which executescript() above has
    # just brought to SCHEMA_VERSION, so it is set rather than preserved; leaving it
    # at whatever version created the file made it report v4 on a v5 schema.
    for key, value in (
        ("embed_model", config.MODEL_NAME),
        ("embed_dim", str(config.EMBED_DIM)),
    ):
        con.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))
    set_meta(con, "schema_version", str(SCHEMA_VERSION))


def get_meta(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def embedding_mismatch(con: sqlite3.Connection) -> str | None:
    """Return a human-readable reason the stored vectors are unusable, else None."""
    model = get_meta(con, "embed_model")
    dim = get_meta(con, "embed_dim")
    if model != config.MODEL_NAME:
        return (
            f"stored embeddings were built with '{model}', "
            f"server is configured for '{config.MODEL_NAME}' — re-embed required"
        )
    if dim != str(config.EMBED_DIM):
        return (
            f"stored embedding dimension is {dim}, "
            f"server is configured for {config.EMBED_DIM} — re-embed required"
        )
    return None
