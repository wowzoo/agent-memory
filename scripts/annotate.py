"""List memories missing graph annotations, and attach annotations from a JSON file.

Two halves of one routine. Editing a hot section changes its body hash, so it gets a
new memory_id and loses its annotations by design (see hot.py) -- which means the
store drifts back toward unannotated on every session that touches a memory file.
Since annotations now feed the recall graph lane, that drift costs recall quality.

    python scripts/annotate.py --list
    python scripts/annotate.py batch.json --dry-run
    python scripts/annotate.py batch.json

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

UNANNOTATED = """
SELECT id, tier, topic, path FROM memories
WHERE id NOT IN (SELECT memory_id FROM memory_entities)
  -- MEMORY.md's index section is pointers to other memories with no facts of its
  -- own; annotating it only duplicates their entities. Skipped so it does not sit
  -- in this list forever.
  AND COALESCE(path, '') NOT LIKE '%MEMORY.md'
ORDER BY tier, topic
"""


def list_unannotated(con) -> int:
    rows = con.execute(UNANNOTATED).fetchall()
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


def apply(con, batch: list, dry_run: bool) -> int:
    entities = relations = 0
    for item in batch:
        row = resolve(con, item)
        if row is None:
            return 1
        if dry_run:
            # `graph.attach` commits, so there is no transaction left to roll back.
            # Resolution is the only failure this can have, and it just ran.
            n_e, n_r = len(item.get("entities") or []), len(item.get("relations") or [])
        else:
            result = graph.attach(
                con, row["id"], item.get("entities"), item.get("relations")
            )
            if result.get("status") == "error":
                print(f"  ! {row['id']}: {result['message']}")
                return 1
            n_e, n_r = result["entities"], result["relations"]
        entities += n_e
        relations += n_r
        print(f"  {row['id'][:8]}  {n_e:>3}e {n_r:>3}r  [{row['tier']}] {row['topic'][:52]}")

    verb = "would attach" if dry_run else "attached"
    print(f"\n{verb} {entities} entities, {relations} relations "
          f"across {len(batch)} memories")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=Path, help="JSON batch to attach")
    parser.add_argument("--list", action="store_true",
                        help="list memories with no annotations and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve every entry and report counts without writing")
    args = parser.parse_args()
    if bool(args.list) == bool(args.batch):
        parser.error("give either --list or a batch file")

    con = db.connect()
    try:
        if args.list:
            return list_unannotated(con)
        batch = json.loads(args.batch.read_text(encoding="utf-8"))
        return apply(con, batch, args.dry_run)
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
