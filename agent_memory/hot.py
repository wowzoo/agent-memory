"""The hot tier: Claude Code's file memory, indexed alongside the archive.

MARM only ever saw its own database. But the authoritative memory for this setup is
the markdown under `~/.claude/projects/<slug>/memory/` -- those files are loaded into
context every session and are edited by hand, so they are the source of truth and the
database is the archive. A recall that searches only the archive answers from stale
copies of facts the files have since corrected.

So hot files are indexed into the same tables as archived memories (one FTS index,
one chunk table, one fusion) and tagged `tier='hot'`. Results carry their tier and
their path.

The tier is provenance, not a score: nothing here ranks a hot row above a cold one.
A flat boost for hot was tried and measured on this corpus and it was worse -- it
promoted files even for questions the files do not answer, taking the fixture's cold
targets to R@1 0% / R@5 40%. Ranking is relevance alone, and `MIN_TIER_SLOTS` only
guarantees that neither tier disappears from the response. See the table in
`docs/measurements.md`.

This tier is read-only here: `mem_write` never writes into the files. They are
maintained by the agent's own memory tooling; we mirror them for retrieval.
"""

import hashlib
import os
import re
import sqlite3
import uuid
from pathlib import Path

from . import embedding, korean

_NAMESPACE = uuid.UUID("2c9d5b71-3e84-4a16-9f0d-7b6e1c8a4d52")

# Claude Code derives its per-project directory by replacing '/' and '_' with '-'
# in the absolute project path (verified against the 12 existing project dirs, e.g.
# /Users/wowzoo/Development/ai-dlc/_accounts/aximo ->
# -Users-wowzoo-Development-ai-dlc--accounts-aximo).
_SLUG_CHARS = re.compile(r"[/_]")

# Split on markdown headings up to h3; deeper headings stay inside their section.
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)

# YAML frontmatter is metadata, not memory. It is stripped before sectioning for two
# reasons: `originSessionId` hex and ISO timestamps are pure noise in the lexical
# index, and `modified:` is rewritten on every edit -- with it in the body, the first
# section of a file hashed differently every time the file was touched, so its graph
# annotations died even when nothing it says had changed.
_FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n", re.DOTALL)


def default_hot_dir() -> Path | None:
    slug = _SLUG_CHARS.sub("-", str(Path(os.getcwd()).resolve()))
    path = Path.home() / ".claude" / "projects" / slug / "memory"
    return path if path.is_dir() else None


def hot_dirs() -> list[Path]:
    """Configured hot directories, else the current project's Claude Code memory."""
    configured = os.environ.get("AGENT_MEMORY_HOT_PATHS", "").strip()
    if configured:
        return [Path(p).expanduser() for p in configured.split(os.pathsep) if p.strip()]
    found = default_hot_dir()
    return [found] if found else []


def _files() -> list[Path]:
    return sorted(
        {p for directory in hot_dirs() for p in directory.glob("*.md") if p.is_file()}
    )


def _sections(text: str, fallback_topic: str) -> list[tuple[str, str]]:
    """(heading, body) pairs. A file with no headings yields one section.

    Sectioning rather than whole-file indexing because these files are long -- 61KB
    at the top end -- and a single vector over 20,000 words locates nothing.
    """
    text = _FRONTMATTER.sub("", text)
    matches = list(_HEADING.finditer(text))
    if not matches:
        stripped = text.strip()
        return [(fallback_topic, stripped)] if stripped else []

    sections = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((fallback_topic, preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        heading = match.group(2).strip()
        if body:
            sections.append((heading, f"{heading}\n\n{body}"))
    return sections


def _digest(path: Path) -> tuple[float, int, str]:
    data = path.read_bytes()
    stat = path.stat()
    return stat.st_mtime, stat.st_size, hashlib.sha256(data).hexdigest()


def _drop(con, path: str) -> None:
    """Remove a file's mirror completely.

    The FTS delete is not optional: memories_fts is a standalone table, so a memory
    row can be deleted while its index entry survives. memory_chunks goes away via
    ON DELETE CASCADE (PRAGMA foreign_keys is on).
    """
    con.execute(
        "DELETE FROM memories_fts WHERE memory_id IN "
        "(SELECT id FROM memories WHERE tier = 'hot' AND path = ?)",
        (path,),
    )
    con.execute("DELETE FROM memories WHERE tier = 'hot' AND path = ?", (path,))
    con.execute("DELETE FROM hot_files WHERE path = ?", (path,))


def _section_id(path: Path, heading: str, body: str) -> str:
    """Identity of a mirrored section: its file, its heading, and its exact body.

    The body hash is in the key on purpose. Graph annotations hang off memory_id with
    ON DELETE CASCADE, so an id that outlives an edit would re-attach yesterday's
    entities to text they were not written about; an id that changes with the body
    makes the edited section a different memory and drops its annotations, which is
    the honest outcome. Position is deliberately *not* in the key -- it was, and that
    made inserting one section renumber every section below it, so an untouched
    paragraph lost its annotations because something above it moved.
    """
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return uuid.uuid5(_NAMESPACE, f"{path}|{heading}|{digest}").hex


def _index_file(
    con, path: Path, project: str | None, agent: str | None
) -> tuple[int, int]:
    """Mirror one file, touching only the sections that actually changed.

    Returns (sections, kept). These files are edited every session, so re-inserting
    the whole file would take the graph annotations of every untouched section with
    it. Unchanged sections are also not re-embedded, which is where the time goes.

    Two sections with an identical heading *and* an identical body collapse into one
    row -- they are the same text, and one index entry answers for both.
    """
    mtime, size, digest = _digest(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _sections(text, path.stem)

    wanted = {_section_id(path, heading, body): (heading, body)
              for heading, body in sections}
    existing = {
        row["id"]
        for row in con.execute(
            "SELECT id FROM memories WHERE tier = 'hot' AND path = ?", (str(path),)
        )
    }
    gone = list(existing - wanted.keys())
    if gone:
        marks = ",".join("?" * len(gone))
        con.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({marks})", gone)
        con.execute(f"DELETE FROM memories WHERE id IN ({marks})", gone)

    for memory_id, (heading, body) in wanted.items():
        if memory_id in existing:
            continue
        con.execute(
            "INSERT INTO memories (id, session, project, agent, topic, content, "
            "created_at, source, tier, path) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime(?, 'unixepoch'), 'hot', 'hot', ?)",
            (memory_id, path.stem, project, agent, heading[:120], body, mtime,
             str(path)),
        )
        con.execute(
            "INSERT INTO memories_fts(topic, body, memory_id) VALUES (?, ?, ?)",
            (korean.index_text(heading), korean.index_text(body), memory_id),
        )
        chunks = embedding.passages(heading, body)
        vectors = embedding.encode_passages(chunks)
        con.executemany(
            "INSERT INTO memory_chunks(memory_id, idx, text, embedding) VALUES (?,?,?,?)",
            [
                (memory_id, i, chunk_text, embedding.to_blob(vec))
                for i, (chunk_text, vec) in enumerate(zip(chunks, vectors))
            ],
        )

    con.execute(
        "INSERT OR REPLACE INTO hot_files(path, mtime, size, hash, project, agent) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (str(path), mtime, size, digest, project, agent),
    )
    return len(wanted), len(existing & wanted.keys())


def _state(con) -> tuple[dict, dict]:
    """What the mirror holds: per-file (mtime, size, hash), and per-directory owner."""
    known = {
        row["path"]: (row["mtime"], row["size"], row["hash"])
        for row in con.execute("SELECT path, mtime, size, hash FROM hot_files")
    }
    # Which project and which agent a directory belongs to, as recorded by whoever
    # indexed it first. Without this both come from the current process, which is wrong
    # for any directory it does not own -- measured for `project`: indexing another
    # project's memory dir relabelled its sections as ours. `agent` is the same fact
    # about the same directory, so it is inherited the same way; a Codex recall that
    # happens to mirror Claude's memory files must not claim them as Codex's.
    # Accumulated per field rather than built as a comprehension: a directory holds many
    # files and only some of their rows carry a name, so the last row read must not be
    # allowed to overwrite a known owner with a NULL.
    owner: dict[str, tuple[str | None, str | None]] = {}
    for row in con.execute("SELECT path, project, agent FROM hot_files"):
        directory = row["path"].rsplit("/", 1)[0]
        known_project, known_agent = owner.get(directory, (None, None))
        owner[directory] = (known_project or row["project"], known_agent or row["agent"])
    return known, owner


def _dirty(known: dict, digests: dict, directories: set[str]) -> bool:
    """Whether `sync` has anything to write: a changed or new file, or a vanished one
    inside a directory this process mirrors (the same rule the removal loop applies)."""
    if any(known.get(key, (None, None, None))[2] != digest
           for key, digest in digests.items()):
        return True
    return any(key not in digests and str(Path(key).parent) in directories
               for key in known)


def sync(con, force: bool = False) -> dict:
    """Bring the hot mirror in line with the files on disk.

    Called before every recall. The cheap part -- stat + hash of a few dozen files --
    runs every time; embedding only runs for sections whose content actually changed,
    so a warm store adds milliseconds rather than re-encoding hundreds of sections.

    `force` means rebuild, not just re-check: it drops each file's mirror so every
    section is embedded again, which is what a model swap needs. Without it, sections
    that still hash the same are left alone, vectors and graph annotations included.
    """
    digests = {}
    for path in _files():
        try:
            digests[str(path)] = _digest(path)[2]
        except OSError:
            continue
    directories = {str(d) for d in hot_dirs()}
    stats = {"indexed": 0, "sections": 0, "kept": 0, "removed": 0, "unchanged": 0}

    known, owner = _state(con)
    if not force and not _dirty(known, digests, directories):
        stats["unchanged"] = len(digests)
        return stats

    # Every harness on a project mirrors the same directory, and section ids are
    # deterministic, so two processes that both see a file changed compute the same
    # ids. Deciding what to write from a read taken outside the write lock let the
    # second one insert ids the first had just committed, and its whole recall died
    # with IntegrityError (reproduced with two processes, 2026-09-24). So take the lock
    # first and decide again under it: a process that waited finds the other's work
    # already there and has nothing left to do. Only when there is work -- a sync with
    # nothing to do never takes the lock, so the common recall stays read-only.
    opened = False
    if not con.in_transaction:
        try:
            con.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            # Another process has held the lock past the busy timeout, which a full
            # first index of a large directory can do. It is mirroring these same
            # files, so answer from the mirror as it stands rather than failing the
            # recall; the next one catches up.
            if "locked" not in str(exc):
                raise
            stats["busy"] = True
            return stats
        opened = True
        known, owner = _state(con)

    project = agent = None
    for key, digest in digests.items():
        path = Path(key)
        cached = known.pop(key, None)
        if not force and cached and cached[2] == digest:
            stats["unchanged"] += 1
            continue
        if project is None:
            from .store import detect_agent, detect_project

            project = detect_project()
            agent = detect_agent()
        if force:
            _drop(con, key)
        known_project, known_agent = owner.get(str(path.parent), (None, None))
        sections, kept = _index_file(
            con, path, known_project or project, known_agent or agent
        )
        stats["sections"] += sections
        stats["kept"] += kept
        stats["indexed"] += 1

    for stale in known:  # files that disappeared or moved
        # Only within the hot directories this process mirrors. `hot_files` is shared
        # by every project registered against this store, so anything left in `known`
        # is either gone or simply someone else's -- and dropping someone else's takes
        # its graph annotations down with it (memory_id CASCADE), which no re-index
        # restores. Measured before this guard: alternating recalls between two
        # projects wiped each other's hot sections and annotations every single sync.
        # The cost of being wrong in this direction is a stale row when a hot
        # directory itself disappears, which is far cheaper than deleting hand-written
        # annotations, and `_index_file` overwrites such a row if the file returns.
        if str(Path(stale).parent) not in directories:
            continue
        _drop(con, stale)
        stats["removed"] += 1

    if stats["indexed"] or stats["removed"] or opened:
        con.commit()
    return stats
