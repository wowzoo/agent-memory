"""The hot tier: Claude Code's file memory, indexed alongside the archive.

MARM only ever saw its own database. But the authoritative memory for this setup is
the markdown under `~/.claude/projects/<slug>/memory/` -- those files are loaded into
context every session and are edited by hand, so they are the source of truth and the
database is the archive. A recall that searches only the archive answers from stale
copies of facts the files have since corrected.

So hot files are indexed into the same tables as archived memories (one FTS index,
one chunk table, one fusion) and tagged `tier='hot'`. Results carry their tier, and
hot rows get a scoring boost, which is how the "files win on overlap" rule stops
being something the agent has to remember and becomes something the tool enforces.

This tier is read-only here: `mem_write` never writes into the files. They are
maintained by the agent's own memory tooling; we mirror them for retrieval.
"""

import hashlib
import os
import re
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


def _index_file(con, path: Path, project: str | None) -> int:
    mtime, size, digest = _digest(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _sections(text, path.stem)

    for index, (heading, body) in enumerate(sections):
        memory_id = uuid.uuid5(_NAMESPACE, f"{path}|{index}|{heading}").hex
        con.execute(
            "INSERT INTO memories"
            "(id, session, project, topic, content, created_at, source, tier, path) "
            "VALUES (?, ?, ?, ?, ?, datetime(?, 'unixepoch'), 'hot', 'hot', ?)",
            (memory_id, path.stem, project, heading[:120], body, mtime, str(path)),
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
        "INSERT INTO hot_files(path, mtime, size, hash) VALUES (?, ?, ?, ?)",
        (str(path), mtime, size, digest),
    )
    return len(sections)


def sync(con, force: bool = False) -> dict:
    """Bring the hot mirror in line with the files on disk.

    Called before every recall. The cheap part -- stat + hash of a few dozen files --
    runs every time; embedding only runs for files whose content actually changed, so
    a warm store adds milliseconds rather than re-encoding hundreds of sections.
    """
    known = {
        row["path"]: (row["mtime"], row["size"], row["hash"])
        for row in con.execute("SELECT path, mtime, size, hash FROM hot_files")
    }
    present = _files()
    project = None
    stats = {"indexed": 0, "sections": 0, "removed": 0, "unchanged": 0}

    for path in present:
        key = str(path)
        try:
            _, _, digest = _digest(path)
        except OSError:
            continue
        cached = known.pop(key, None)
        if not force and cached and cached[2] == digest:
            stats["unchanged"] += 1
            continue
        if project is None:
            from .store import detect_project

            project = detect_project()
        _drop(con, key)
        stats["sections"] += _index_file(con, path, project)
        stats["indexed"] += 1

    for stale in known:  # files that disappeared or moved
        _drop(con, stale)
        stats["removed"] += 1

    if stats["indexed"] or stats["removed"]:
        con.commit()
    return stats
