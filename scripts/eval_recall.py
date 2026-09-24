"""Known-item retrieval eval against the live store, through the real recall path.

For each stored memory, a sentence from the middle of its body becomes the query
and the memory itself is the single correct answer. No hand labelling needed, and
because it calls `store.recall` rather than reimplementing scoring, it measures the
code that actually serves `mem_recall`.

Reports each lane separately so a regression can be attributed:

    python scripts/eval_recall.py
    python scripts/eval_recall.py --project aidlc-v2-patch
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, store  # noqa: E402

DEPTH = 20  # ranks deeper than this are scored as a miss


def middle_sentence(content: str) -> str | None:
    body = content.split("\n", 1)[-1] if "\n" in content else content
    parts = [s.strip() for s in re.split(r"(?<=[다\.])\s+", body) if len(s.strip()) > 30]
    return parts[len(parts) // 2] if parts else None


def run(mode: str, cases: list[tuple[str, str]], project: str | None,
        rerank: bool | None = False, window: int = DEPTH) -> None:
    con = db.connect()
    ranks, skipped = [], 0
    for memory_id, query in cases:
        response = store.recall(
            con, query, limit=window, project=project, detail=1, mode=mode,
            rerank=rerank,
        )
        skipped += bool(response.get("rerank_skipped"))
        ids = [r["id"] for r in response["results"]]
        ranks.append(ids.index(memory_id) + 1 if memory_id in ids else window + 1)
    con.close()

    n = len(ranks)
    at1 = sum(r == 1 for r in ranks) / n
    at5 = sum(r <= 5 for r in ranks) / n
    mrr = sum(1 / r for r in ranks) / n
    print(f"  {mode:9} R@1={at1:6.1%}  R@5={at5:6.1%}  MRR={mrr:.3f}  "
          f"median={int(statistics.median(ranks))}"
          + (f"  gate skipped {skipped}/{n}" if rerank is None else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of CASES, not the result window (--window)")
    parser.add_argument("--window", type=int, default=DEPTH,
                        help=f"the recall limit a caller would pass (default {DEPTH}). "
                             "Comparing MRR across windows is meaningless: a miss scores "
                             "1/(window+1), so 1/21 here and 1/6 at --window 5")
    parser.add_argument("--rerank", choices=("off", "on", "auto"), default="off",
                        help="off = never (every number this fixture has produced), "
                             "on = forced past the gate, auto = the production path "
                             "(config default, gate included)")
    args = parser.parse_args()

    con = db.connect()
    sql = "SELECT id, content FROM memories"
    params: list = []
    if args.project:
        sql += " WHERE project = ?"
        params.append(args.project)
    # Ordered, so that --limit takes the same cases every run. Without it the slice
    # depends on whatever order SQLite happens to return and the fixture is not fixed.
    sql += " ORDER BY id"
    rows = con.execute(sql, params).fetchall()
    con.close()

    cases = []
    for row in rows:
        query = middle_sentence(row["content"])
        if query:
            cases.append((row["id"], query))
    if args.limit:
        cases = cases[: args.limit]

    print(f"{len(cases)} cases (of {len(rows)} memories)"
          + (f", project={args.project}" if args.project else ""))
    rerank = {"off": False, "on": True, "auto": None}[args.rerank]
    print(f"  window={args.window}, rerank={args.rerank}")
    modes = ("lexical", "semantic", "hybrid") if rerank is False else ("hybrid",)
    for mode in modes:
        run(mode, cases, args.project, rerank, args.window)


if __name__ == "__main__":
    main()
