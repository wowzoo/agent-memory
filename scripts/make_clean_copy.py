"""Build a store copy with this round's own evaluation records removed.

Measurement hygiene ① in the README: this repository's hot tier indexes its own project
memory file, so the moment a session writes down "the miss list was these queries", BM25
starts answering those queries with the *record* instead of the memory. Every ranking A/B
since 2026-08-30 has therefore been measured on a copy with that day's sections dropped
-- by hand, twice, which is how a hygiene rule quietly stops being followed.

    python scripts/make_clean_copy.py                       # today's sections
    python scripts/make_clean_copy.py --date 2026-08-30
    AGENT_MEMORY_DB=/tmp/agent-memory-clean.db python scripts/eval_hot.py --limit 5

Only this project's own records are touched. Another project's sections written the same
day are not contamination -- they are corpus, and removing them would move the baseline
(which is its own recorded mistake: syncing someone else's hot tier broke the fixture
anchors).

⚠️ Both tiers, which the README's rule did not say. The rule was written about hot
sections because that is where it was caught, but a round also writes cold records with
`mem_write`, and those bury the answer exactly as well: measured 2026-08-30, the archived
record *"2026-08-30 리랭크 축 종결"* sits at rank 2 for the hot fixture's own
"크로스 인코더" query. Dropping hot only leaves that in place.

`cp` is wrong for this DB: WAL mode leaves the log megabytes ahead of the main file, so a
copy made that way silently omits recent writes. `.backup` is the supported path.
"""

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config  # noqa: E402

# The hot file and the cold project whose records are this repository's self-observation.
OWN_MEMORY = str(Path(
    "~/.claude/projects/-Users-wowzoo-Development-agent-memory/memory"
).expanduser())
PROJECT = "agent-memory"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "data"
DEFAULT_OUT = Path("/tmp/agent-memory-clean.db")


def is_gold(topic: str, path: str | None) -> bool:
    """Would deleting this row remove a fixture answer?

    Hot golds key on (path, heading); cold golds on a topic prefix. Both are checked
    because the copy now touches both tiers.
    """
    hot = json.loads((FIXTURES / "hot_queries.json").read_text())["cases"]
    cold = json.loads((FIXTURES / "paraphrase_queries.json").read_text())["cases"]
    if path is not None and any(
        str(Path(c["path"]).expanduser()) == path and c["heading"][:120] == topic
        for c in hot if "path" in c
    ):
        return True
    return any(topic.startswith(c["topic_match"]) for c in cold if "topic_match" in c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="drop this project's hot sections indexed on this date")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    src = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    args.out.unlink(missing_ok=True)
    dst = sqlite3.connect(args.out)
    src.backup(dst)
    src.close()
    dst.row_factory = sqlite3.Row

    before = dst.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    doomed = dst.execute(
        "SELECT id, topic, path, tier FROM memories WHERE created_at LIKE ? AND "
        "((tier = 'hot' AND path LIKE ?) OR (tier = 'cold' AND project = ?))",
        (f"{args.date}%", f"{OWN_MEMORY}%", PROJECT),
    ).fetchall()

    # A gold that is also a record of this round is unresolvable either way: keeping it
    # lets the record answer its own query, dropping it removes the answer. Refuse rather
    # than shrink the denominator silently -- `cases_of` skips what it cannot resolve.
    clash = [r for r in doomed if is_gold(r["topic"], r["path"])]
    if clash:
        print("REFUSING: these sections are also hot fixture golds --")
        for r in clash:
            print(f"  {r['topic'][:70]}")
        dst.close()
        sys.exit(1)

    ids = [r["id"] for r in doomed]
    if ids:
        marks = ",".join("?" * len(ids))
        for sql in (
            f"DELETE FROM memories_fts WHERE memory_id IN ({marks})",
            f"DELETE FROM memory_chunks WHERE memory_id IN ({marks})",
            f"DELETE FROM memory_entities WHERE memory_id IN ({marks})",
            f"DELETE FROM relations WHERE memory_id IN ({marks})",
            f"DELETE FROM memories WHERE id IN ({marks})",
        ):
            dst.execute(sql, ids)
        dst.commit()

    after = dst.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
    print(f"{args.out}  memories {before} -> {after} "
          f"(dropped {len(ids)}), integrity_check={ok}")
    for r in doomed:
        print(f"  - [{r['tier']:<4}] {r['topic'][:68]}")
    dst.close()


if __name__ == "__main__":
    main()
