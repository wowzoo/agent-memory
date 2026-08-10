"""List memories missing graph annotations, and attach annotations from a JSON file.

Two halves of one routine. Editing a hot section changes its body hash, so it gets a
new memory_id and loses its annotations by design (see hot.py) -- which means the
store drifts back toward unannotated on every session that touches a memory file.
Since annotations now feed the recall graph lane, that drift costs recall quality.

    python scripts/annotate.py --list
    python scripts/annotate.py batch.json --dry-run
    python scripts/annotate.py batch.json
    python scripts/annotate.py batch.json --append   # 기존 주석에 더한다

⚠️ Attaching REPLACES a memory's annotations. A batch carrying one entity takes a
memory that had seven down to one -- measured, and it cost real work. Adding to what
is there needs `--append`, and every line of output now prints what was there before.

Annotations go through a file rather than `mem_annotate` arguments: Korean sent as
MCP tool arguments is corrupted about a third of the time (measured: 94 of 272 entity
names across 8 batches), and a file round-trip preserves it. The batch is a JSON list of

    {"memory_id": "3f68be58", "entities": [...], "relations": [...]}

resolved by id prefix, falling back to an exact hot `topic` match so a batch written
before an edit can still be applied afterwards.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, graph  # noqa: E402
from agent_memory.graph import normalize  # noqa: E402

UNANNOTATED = """
SELECT id, tier, topic, path FROM memories
WHERE id NOT IN (SELECT memory_id FROM memory_entities)
  -- MEMORY.md's index section is pointers to other memories with no facts of its
  -- own; annotating it only duplicates their entities. Skipped so it does not sit
  -- in this list forever.
  AND COALESCE(path, '') NOT LIKE '%MEMORY.md'
  AND (:project IS NULL OR project = :project)
ORDER BY tier, topic
"""


def list_unannotated(con, project: str | None) -> int:
    rows = con.execute(UNANNOTATED, {"project": project}).fetchall()
    for row in rows:
        print(f"{row['id'][:8]}  [{row['tier']}] {row['topic']}")
    print(f"\n{len(rows)} unannotated")
    return 0


def resolve(con, item: dict):
    prefix = item.get("memory_id", "")
    rows = con.execute(
        "SELECT id, topic, tier FROM memories WHERE substr(id, 1, ?) = ?",
        (len(prefix), prefix),
    ).fetchall() if prefix else []
    if not rows and item.get("topic"):
        rows = con.execute(
            "SELECT id, topic, tier FROM memories WHERE tier = 'hot' AND topic = ?",
            (item["topic"],),
        ).fetchall()
    if len(rows) != 1:
        print(f"  ! {prefix or item.get('topic')} matched {len(rows)} memories")
        return None
    return rows[0]


def current(con, memory_id: str) -> tuple[list, list]:
    """What is already attached, as batch-shaped dicts."""
    entities = [
        {"name": display, "type": etype}
        for display, etype in con.execute(
            "SELECT e.display, e.type FROM memory_entities me "
            "JOIN entities e ON e.name = me.entity WHERE me.memory_id = ?",
            (memory_id,),
        )
    ]
    relations = [
        {"source": s, "predicate": p, "target": t}
        for s, p, t in con.execute(
            "SELECT es.display, r.predicate, et.display FROM relations r "
            "JOIN entities es ON es.name = r.source "
            "JOIN entities et ON et.name = r.target WHERE r.memory_id = ?",
            (memory_id,),
        )
    ]
    return entities, relations


def merged(con, memory_id: str, item: dict) -> tuple[list, list]:
    """Union of what is attached and what the batch supplies, batch winning on type."""
    have_e, have_r = current(con, memory_id)
    new_e = list(item.get("entities") or [])
    new_r = list(item.get("relations") or [])
    keys = {normalize(e["name"] if isinstance(e, dict) else str(e)) for e in new_e}
    entities = new_e + [e for e in have_e if normalize(e["name"]) not in keys]
    seen = {
        (normalize(r.get("source", "")), (r.get("predicate") or "").strip().lower(),
         normalize(r.get("target", "")))
        for r in new_r if isinstance(r, dict)
    }
    relations = new_r + [
        r for r in have_r
        if (normalize(r["source"]), r["predicate"], normalize(r["target"])) not in seen
    ]
    return entities, relations


def apply(con, batch: list, dry_run: bool, append: bool) -> int:
    entities = relations = 0
    for item in batch:
        row = resolve(con, item)
        if row is None:
            return 1
        had_e, had_r = current(con, row["id"])
        want_e, want_r = (
            merged(con, row["id"], item) if append
            else (item.get("entities"), item.get("relations"))
        )
        if dry_run:
            # `graph.attach` commits, so there is no transaction left to roll back.
            # Resolution is the only failure this can have, and it just ran.
            n_e, n_r = len(want_e or []), len(want_r or [])
        else:
            result = graph.attach(con, row["id"], want_e, want_r)
            if result.get("status") == "error":
                print(f"  ! {row['id']}: {result['message']}")
                return 1
            n_e, n_r = result["entities"], result["relations"]
        entities += n_e
        relations += n_r
        # Both sides of the change, because attach REPLACES a memory's annotations
        # rather than adding to them. A batch holding one entity silently took a
        # memory from seven entities to two, and a bare "2e" gave no sign of it.
        arrow = "" if (len(had_e), len(had_r)) == (n_e, n_r) else \
                f"  (was {len(had_e)}e {len(had_r)}r)"
        loss = "  ⚠ fewer than before" if n_e < len(had_e) and not append else ""
        print(f"  {row['id'][:8]}  {n_e:>3}e {n_r:>3}r{arrow}{loss}  "
              f"[{row['tier']}] {row['topic'][:44]}")

    verb = "would attach" if dry_run else "attached"
    mode = " (merged with existing)" if append else " (replacing existing)"
    print(f"\n{verb} {entities} entities, {relations} relations "
          f"across {len(batch)} memories{mode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=Path, help="JSON batch to attach")
    parser.add_argument("--list", action="store_true",
                        help="list memories with no annotations and exit")
    # Several projects share one store, so an unfiltered list is mostly other
    # people's work. A per-project routine needs to see only its own.
    parser.add_argument("--project", default=None,
                        help="restrict --list to one project (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve every entry and report counts without writing")
    parser.add_argument("--append", action="store_true",
                        help="merge with a memory's existing annotations instead of "
                             "replacing them")
    args = parser.parse_args()
    if bool(args.list) == bool(args.batch):
        parser.error("give either --list or a batch file")

    con = db.connect()
    try:
        if args.list:
            return list_unannotated(con, args.project)
        batch = json.loads(args.batch.read_text(encoding="utf-8"))
        return apply(con, batch, args.dry_run, args.append)
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
