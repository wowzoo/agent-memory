"""Rename corrupted entity names from a verified bad->good mapping file.

Korean entity names arrive through MCP arguments escaped, and re-typing them by
hand corrupts a fresh set every batch (`전체`->`전재`, `반응성`->`반융성`). The only
approach that converges is to settle on the correct strings in a file, then rename
at the DB level in one pass -- which is what this does.

The mapping is a TSV of `bad<TAB>good`, one pair per line; blank lines are skipped.
Names are matched on `graph.normalize` keys, so case and spacing need not match.

    python scripts/fix_entity_names.py entity_rename.tsv
    python scripts/fix_entity_names.py entity_rename.tsv --dry-run

A copy of the database is written next to it as `*.db.pre-rename` before any write.
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config  # noqa: E402
from agent_memory.graph import normalize  # noqa: E402


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not all(f.strip() for f in fields):
            raise SystemExit(f"{path}:{lineno}: expected 'bad<TAB>good', got {line!r}")
        pairs.append((fields[0], fields[1]))
    return pairs


def rename(con: sqlite3.Connection, pairs: list[tuple[str, str]]) -> None:
    for bad, good in pairs:
        bk, gk = normalize(bad), normalize(good)
        if not con.execute("SELECT 1 FROM entities WHERE name = ?", (bk,)).fetchone():
            print(f"SKIP (not found): {bad}")
            continue
        if con.execute("SELECT 1 FROM entities WHERE name = ?", (gk,)).fetchone():
            # The good name is already an entity, so the rows have to be folded into
            # it and the duplicates dropped -- UPDATE alone would collide on the PK.
            con.execute("UPDATE OR IGNORE memory_entities SET entity = ? WHERE entity = ?", (gk, bk))
            con.execute("UPDATE OR IGNORE relations SET source = ? WHERE source = ?", (gk, bk))
            con.execute("UPDATE OR IGNORE relations SET target = ? WHERE target = ?", (gk, bk))
            con.execute("DELETE FROM memory_entities WHERE entity = ?", (bk,))
            con.execute("DELETE FROM relations WHERE source = ? OR target = ?", (bk, bk))
            con.execute("DELETE FROM entities WHERE name = ?", (bk,))
            print(f"MERGED into existing: {bad} -> {good}")
        else:
            con.execute("UPDATE entities SET name = ?, display = ? WHERE name = ?", (gk, good, bk))
            con.execute("UPDATE memory_entities SET entity = ? WHERE entity = ?", (gk, bk))
            con.execute("UPDATE relations SET source = ? WHERE source = ?", (gk, bk))
            con.execute("UPDATE relations SET target = ? WHERE target = ?", (gk, bk))
            print(f"RENAMED: {bad} -> {good}")


def verify(con: sqlite3.Connection) -> int:
    dangling = con.execute(
        "SELECT COUNT(*) FROM memory_entities WHERE entity NOT IN (SELECT name FROM entities)"
    ).fetchone()[0]
    dangling += con.execute(
        "SELECT COUNT(*) FROM relations WHERE source NOT IN (SELECT name FROM entities)"
        " OR target NOT IN (SELECT name FROM entities)"
    ).fetchone()[0]
    fk_rows = len(con.execute("PRAGMA foreign_key_check").fetchall())
    print(f"dangling refs: {dangling}")
    print("integrity_check:", con.execute("PRAGMA integrity_check").fetchone()[0])
    print("foreign_key_check rows:", fk_rows)
    return dangling + fk_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename corrupted entity names from a mapping file.")
    parser.add_argument("mapping", type=Path, help="TSV of bad<TAB>good entity names")
    parser.add_argument(
        "--db", type=Path, default=config.DB_PATH, help="store to rewrite (default: the configured store)"
    )
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = parser.parse_args()

    pairs = read_pairs(args.mapping)
    print(f"{len(pairs)} pair(s) from {args.mapping}")

    if not args.dry_run:
        backup = args.db.with_suffix(".db.pre-rename")
        shutil.copy(args.db, backup)
        print(f"backup: {backup}")

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("BEGIN")
    # Renames touch parent and child rows in the same transaction, so the FK checks
    # have to hold off until COMMIT rather than firing per statement.
    con.execute("PRAGMA defer_foreign_keys = ON")
    try:
        rename(con, pairs)
        if args.dry_run:
            con.rollback()
            print("dry run -- rolled back")
        else:
            con.commit()
        problems = verify(con)
    finally:
        con.close()

    if problems:
        raise SystemExit("verification failed -- restore from the backup")


if __name__ == "__main__":
    main()
