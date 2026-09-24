"""Take back the annotations `prune_moved_hot.py` moved onto a *different* body.

That script matched a dead section to its live twin on (file name, heading) and moved
the graph annotations across. For 1,111 of 1,132 sections the two bodies were byte
identical, so the move was a no-op in meaning. For 21 they were not: the file had been
edited after the move, so the dead row held an older body and the "twin" was a
different text with the same heading. Measured 2026-09-13 against the pre-prune backup,
five of those had annotations -- 35 entity links and 27 relations -- and they are now
attached to a body they were never written about.

That is backwards from how a section's identity works here: the id is
uuid5(path|heading|body), precisely so that editing a body drops the annotations
instead of inheriting them (`hot._index_file`). Re-annotating the new body is a session's
job; carrying the old links over is not.

Only links that (a) the dead row had, (b) the twin did not have before the prune, and
(c) the twin has now are removed. A link the twin already carried is its own, and one
the annotator added since is not ours to touch.

    python scripts/undo_bad_transplants.py ~/.agent-memory/memory.db.bak-...-pathprune
    python scripts/undo_bad_transplants.py <backup> --apply
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db  # noqa: E402


def bad_transplants(before: sqlite3.Connection, live: sqlite3.Connection):
    """[(twin_id, heading, entities, relations)] that came from a different body."""
    twins_by_key = defaultdict(list)
    dead = []
    for row in before.execute(
        "SELECT id, path, topic, content FROM memories "
        "WHERE tier = 'hot' AND path IS NOT NULL"
    ):
        key = (Path(row["path"]).name, row["topic"])
        if Path(row["path"]).exists():
            twins_by_key[key].append(row)
        else:
            dead.append((row, key))

    out = []
    for row, key in dead:
        twins = twins_by_key.get(key, [])
        if len(twins) != 1 or twins[0]["content"] == row["content"]:
            continue
        twin = twins[0]["id"]
        old_e = {r[0] for r in before.execute(
            "SELECT entity FROM memory_entities WHERE memory_id = ?", (row["id"],))}
        had_e = {r[0] for r in before.execute(
            "SELECT entity FROM memory_entities WHERE memory_id = ?", (twin,))}
        now_e = {r[0] for r in live.execute(
            "SELECT entity FROM memory_entities WHERE memory_id = ?", (twin,))}
        old_r = {tuple(r) for r in before.execute(
            "SELECT source, predicate, target FROM relations WHERE memory_id = ?", (row["id"],))}
        had_r = {tuple(r) for r in before.execute(
            "SELECT source, predicate, target FROM relations WHERE memory_id = ?", (twin,))}
        now_r = {tuple(r) for r in live.execute(
            "SELECT source, predicate, target FROM relations WHERE memory_id = ?", (twin,))}
        entities = (now_e - had_e) & old_e
        relations = (now_r - had_r) & old_r
        if entities or relations:
            out.append((twin, row["topic"], entities, relations))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("backup", type=Path, help="the pre-prune backup to compare against")
    ap.add_argument("--apply", action="store_true", help="without it, only reports")
    args = ap.parse_args()

    before = sqlite3.connect(f"file:{args.backup}?mode=ro", uri=True)
    before.row_factory = sqlite3.Row
    live = db.connect()

    found = bad_transplants(before, live)
    total_e = sum(len(e) for _, _, e, _ in found)
    total_r = sum(len(r) for _, _, _, r in found)
    print(f"annotations moved from a different body: {total_e} entity links, "
          f"{total_r} relations, across {len(found)} sections")
    for _, heading, entities, relations in found:
        print(f"  {heading[:55]}: -{len(entities)}e -{len(relations)}r")

    if not args.apply:
        print("\nreport only; pass --apply to remove them")
        return
    for twin, _, entities, relations in found:
        for entity in entities:
            live.execute(
                "DELETE FROM memory_entities WHERE memory_id = ? AND entity = ?",
                (twin, entity))
        for source, predicate, target in relations:
            live.execute(
                "DELETE FROM relations WHERE memory_id = ? AND source = ? "
                "AND predicate = ? AND target = ?",
                (twin, source, predicate, target))
    live.commit()
    print(f"\nremoved {total_e} entity links, {total_r} relations")
    print("memory_entities now:", live.execute(
        "SELECT COUNT(*) FROM memory_entities").fetchone()[0])
    print("relations now:", live.execute(
        "SELECT COUNT(*) FROM relations").fetchone()[0])


if __name__ == "__main__":
    main()
