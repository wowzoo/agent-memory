"""Evaluate recall on hand-authored paraphrase queries.

This is the harness that should decide lane weighting. `eval_recall.py` quotes the
target memory verbatim, so it measures lexical overlap the real world never
supplies; these queries are phrased the way the user actually asks.

    python scripts/eval_paraphrase.py
    python scripts/eval_paraphrase.py --show-misses
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, store  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "paraphrase_queries.json"
DEPTH = 20


def resolve(con, topic_prefix: str) -> str | None:
    """Exact prefix comparison, not LIKE: topics contain '_' and '%' (file names,
    percentages), which LIKE would treat as wildcards and match too broadly."""
    rows = con.execute(
        "SELECT id FROM memories WHERE substr(topic, 1, ?) = ?",
        (len(topic_prefix), topic_prefix),
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument(
        "--tier",
        choices=("hot", "cold"),
        default=None,
        help="restrict recall to one tier (default: both, interleaved)",
    )
    args = parser.parse_args()

    cases = json.loads(FIXTURE.read_text())["cases"]
    con = db.connect()

    resolved = []
    for case in cases:
        full = resolve(con, case["topic_match"])
        if full is None:
            print(f"  ! topic_match did not resolve uniquely — skipped: "
                  f"{case['topic_match'][:50]}")
            continue
        resolved.append((full, case["query"]))
    print(f"{len(resolved)} paraphrase cases"
          + (f", tier={args.tier}" if args.tier else ", both tiers") + "\n")

    per_mode = {}
    for mode in ("lexical", "semantic", "hybrid"):
        ranks, misses = [], []
        for memory_id, query in resolved:
            results = store.recall(
                con, query, limit=DEPTH, detail=1, mode=mode, tier=args.tier
            )["results"]
            ids = [r["id"] for r in results]
            rank = ids.index(memory_id) + 1 if memory_id in ids else DEPTH + 1
            ranks.append(rank)
            if rank > 5:
                misses.append((query, rank))
        n = len(ranks)
        per_mode[mode] = ranks
        print(f"  {mode:9} R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
              f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
              f"MRR={sum(1/r for r in ranks)/n:.3f}  "
              f"median={int(statistics.median(ranks))}")
        if args.show_misses and misses:
            for query, rank in misses:
                print(f"      rank {rank:>3}  {query}")

    # Does fusion actually beat its best single lane, case by case?
    best_single = "lexical" if sum(1/r for r in per_mode["lexical"]) >= sum(
        1/r for r in per_mode["semantic"]) else "semantic"
    better = sum(h < s for h, s in zip(per_mode["hybrid"], per_mode[best_single]))
    worse = sum(h > s for h, s in zip(per_mode["hybrid"], per_mode[best_single]))
    print(f"\n  hybrid vs {best_single}: better on {better}, worse on {worse}, "
          f"tied on {len(resolved) - better - worse}")
    con.close()


if __name__ == "__main__":
    main()
