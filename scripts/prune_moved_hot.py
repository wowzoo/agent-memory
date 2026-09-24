"""Retire the hot mirror a project left behind when its directory moved.

`sync()` only drops stale rows inside the hot directories the calling process
mirrors, because dropping another project's rows takes its graph annotations down
with it (`memory_id` CASCADE) and no re-index restores them. That guard is right,
and it has one hole: when a project moves (`~/Development/x` -> `~/workspaces/x`),
its old memory directory never appears in anyone's `hot_dirs()` again, so nothing
ever retires it. Both mirrors then sit in the store.

Measured 2026-09-13, six days after this repository's own move: 1,193 of 2,802 hot
sections (43%) belonged to files no longer on disk. Most were byte-identical copies of a
live twin, but not all -- 26 held a body no live section matched and 8 had no twin at
all, which is what the three-way decision below is for. `memories_fts` is one table
across both tiers and `bm25()` takes IDF and average document length from the whole
table, so those ghosts move even the cold control numbers -- and a fixture gold whose
twin comes back instead scores as a miss.

Deleting alone is not enough: 813 of 1,065 annotations hung off the *dead* rows, because
they were annotated before the move. A section's id is uuid5(path|heading|body), so the
twin at the new path is a different id and does not inherit them. Annotations are
re-attached by hand per session, so where the body is unchanged they are moved here
rather than re-made.

    python scripts/prune_moved_hot.py                 # report, changes nothing
    python scripts/prune_moved_hot.py --apply

A dead section is matched to a live one by file name, heading **and body**, and what
happens to it depends on what the match guarantees:

- a live twin with the same body -> annotations move to it, the dead row goes.
- a twin whose body differs -> the row still goes (the file it mirrored is gone, and a
  stale snapshot answering recalls is the problem this script exists to remove), but
  **its annotations are dropped, not moved.** A section's id is uuid5(path|heading|body)
  exactly so that editing a body loses its annotations instead of inheriting them
  (`hot._index_file`); carrying them onto a different text is backwards.
- no live twin at all -> kept unconditionally. The file is gone from disk and this row
  is the last copy of what it said (measured: two of them are the newest records in
  the store), so deleting it would be the only irreversible thing here.

⚠️ The body comparison is why this reads the way it does. The first version keyed on
(file name, heading) alone and moved annotations across regardless: of 1,132 sections it
retired, 21 had a *different* body -- the file had been edited after the move -- and 5 of
those carried 35 entity links and 27 relations that landed on a text they were never
written about. `scripts/undo_bad_transplants.py` took those back. Matching totals is not
evidence of a correct move: `memory_entities` and `relations` counts were unchanged
through the whole mistake, because the rows were moved rather than lost.
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db  # noqa: E402


def plan(con) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """(move, drop, keep): rows to re-parent, rows to delete outright, rows to leave.

    `drop` is deleted without moving anything, which covers both a body that no live twin
    matches and a duplicate whose twin already carries the annotations.
    """
    live: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    dead_rows = []
    for row in con.execute(
        "SELECT id, path, topic, content FROM memories "
        "WHERE tier = 'hot' AND path IS NOT NULL"
    ):
        key = (Path(row["path"]).name, row["topic"])
        if Path(row["path"]).exists():
            live[key].append(row)
        else:
            dead_rows.append((row, key))

    move, drop, keep = [], [], []
    for row, key in dead_rows:
        twins = live.get(key, [])
        same_body = [t["id"] for t in twins if t["content"] == row["content"]]
        if not twins:
            keep.append((row["id"], "no live twin -- last copy of this text"))
        elif len(same_body) == 1:
            move.append((row["id"], same_body[0]))
        elif same_body:
            drop.append((row["id"], "duplicate"))
        else:
            drop.append((row["id"], "stale body"))
    return move, drop, keep


def count_annotations(con, ids: list[str]) -> tuple[int, int]:
    if not ids:
        return 0, 0
    marks = ",".join("?" * len(ids))
    ents = con.execute(
        f"SELECT COUNT(*) FROM memory_entities WHERE memory_id IN ({marks})", ids
    ).fetchone()[0]
    rels = con.execute(
        f"SELECT COUNT(*) FROM relations WHERE memory_id IN ({marks})", ids
    ).fetchone()[0]
    return ents, rels


def apply(con, matched: list[tuple[str, str]], drop: list[tuple[str, str]]) -> dict:
    """Move annotations to the twin, then delete the dead rows.

    OR IGNORE because the twin may already carry the same entity -- the primary key
    is (memory_id, entity), so a re-annotated twin would otherwise abort the move.
    The FTS delete is explicit for the same reason it is in `hot._drop`: memories_fts
    is a standalone table and does not follow the CASCADE.
    """
    stats = {"entities": 0, "relations": 0, "sections": 0, "files": 0}
    for dead_id, live_id in matched:
        cur = con.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity) "
            "SELECT ?, entity FROM memory_entities WHERE memory_id = ?",
            (live_id, dead_id),
        )
        stats["entities"] += cur.rowcount
        cur = con.execute(
            "INSERT OR IGNORE INTO relations (source, predicate, target, memory_id) "
            "SELECT source, predicate, target, ? FROM relations WHERE memory_id = ?",
            (live_id, dead_id),
        )
        stats["relations"] += cur.rowcount

    ids = [dead for dead, _ in matched] + [dead for dead, _ in drop]
    for start in range(0, len(ids), 500):
        batch = ids[start:start + 500]
        marks = ",".join("?" * len(batch))
        con.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({marks})", batch)
        cur = con.execute(f"DELETE FROM memories WHERE id IN ({marks})", batch)
        stats["sections"] += cur.rowcount

    # A file's mirror row goes only when nothing of it is left, so a partially matched
    # file keeps its digest and is not re-indexed from a path that no longer exists.
    for row in con.execute("SELECT path FROM hot_files").fetchall():
        path = row["path"]
        if Path(path).exists():
            continue
        left = con.execute(
            "SELECT COUNT(*) FROM memories WHERE tier = 'hot' AND path = ?", (path,)
        ).fetchone()[0]
        if not left:
            con.execute("DELETE FROM hot_files WHERE path = ?", (path,))
            stats["files"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="without it, only reports")
    args = ap.parse_args()

    con = db.connect()
    move, drop, keep = plan(con)
    ents, rels = count_annotations(con, [dead for dead, _ in move])
    print(f"re-parent and delete : {len(move)} sections")
    print(f"  annotations to move: {ents} entity links, {rels} relations")
    dup = sum(1 for _, why in drop if why == "duplicate")
    stale = len(drop) - dup
    stale_ann = count_annotations(con, [d for d, why in drop if why == "stale body"])
    print(f"delete, nothing moved: {len(drop)} sections "
          f"({dup} duplicate, {stale} stale body)")
    print(f"  annotations dropped with the stale bodies: "
          f"{stale_ann[0]} entity links, {stale_ann[1]} relations")
    print(f"keep                 : {len(keep)} sections")
    for dead_id, reason in keep:
        row = con.execute("SELECT path, topic FROM memories WHERE id = ?", (dead_id,)).fetchone()
        print(f"  {reason}: {row['topic'][:55]}  <- {Path(row['path']).name}")

    if not args.apply:
        print("\nreport only; pass --apply to move and delete")
        return
    stats = apply(con, move, drop)
    con.commit()
    print(f"\nmoved {stats['entities']} entity links, {stats['relations']} relations")
    print(f"deleted {stats['sections']} sections, {stats['files']} hot_files rows")
    print("hot sections now:", con.execute(
        "SELECT COUNT(*) FROM memories WHERE tier = 'hot'").fetchone()[0])


if __name__ == "__main__":
    main()
