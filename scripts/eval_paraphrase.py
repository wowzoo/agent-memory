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
# `off` rather than "follow the config": every number this harness has ever produced is
# an un-reranked baseline, and reranking became the default on 2026-08-30. Following the
# config would silently redefine the denominator. `auto` is the production path.
RERANK = {"off": False, "on": True, "auto": None}


def resolve(con, topic_prefix: str) -> str | None:
    """Exact prefix comparison, not LIKE: topics contain '_' and '%' (file names,
    percentages), which LIKE would treat as wildcards and match too broadly."""
    rows = con.execute(
        "SELECT id FROM memories WHERE substr(topic, 1, ?) = ?",
        (len(topic_prefix), topic_prefix),
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def resolve_hot(con, path: str, heading: str) -> str | None:
    """A hot section keys on its file and heading, not on a topic prefix.

    Its id is uuid5(path|heading|body-hash), so an id written into a fixture stops
    resolving the next time that section is edited; and headings repeat across files
    ('## 다음에 할 일'), so a prefix alone is not unique. Topics are stored truncated.

    Fixture paths are `~`-relative so the committed file carries no home directory;
    the store's are absolute, so expand before comparing.
    """
    rows = con.execute(
        "SELECT id FROM memories WHERE tier = 'hot' AND path = ? AND topic = ?",
        (str(Path(path).expanduser()), heading[:120]),
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=FIXTURE,
        help="case file (default: the 72 paraphrase queries every A/B so far was "
             "measured on; collected queries live in their own file)",
    )
    parser.add_argument(
        "--tier",
        choices=("hot", "cold"),
        default=None,
        help="restrict recall to one tier (default: both, interleaved)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEPTH,
        help="how many results recall returns. 20 is what every A/B so far used; 5 is "
             "RECALL_LIMIT, the window a caller actually gets -- and the only one where "
             "tier slots and lane rescue can change anything",
    )
    parser.add_argument(
        "--rerank",
        choices=tuple(RERANK),
        default="off",
        help="off (default) is the baseline every past A/B was measured on; on forces "
             "the cross-encoder past the gate (~3 s/query, hybrid only); auto follows "
             "the config, which is what production does",
    )
    args = parser.parse_args()

    cases = json.loads(args.fixture.read_text())["cases"]
    con = db.connect()

    resolved = []
    for case in cases:
        full = (resolve_hot(con, case["path"], case["heading"]) if "path" in case
                else resolve(con, case["topic_match"]))
        if full is None:
            key = case.get("topic_match") or case.get("heading", "")
            print(f"  ! target did not resolve uniquely — skipped: {key[:50]}")
            continue
        resolved.append((full, case["query"]))
    print(f"{len(resolved)} paraphrase cases"
          + (f", tier={args.tier}" if args.tier else ", both tiers") + "\n")

    per_mode = {}
    # With --rerank only hybrid is measured: the reranker reorders whatever the lanes
    # fused, so pairing it with a single-lane baseline measures the baseline, not it.
    modes = ("hybrid",) if args.rerank != "off" else ("lexical", "semantic", "hybrid")
    for mode in modes:
        ranks, misses = [], []
        for memory_id, query in resolved:
            results = store.recall(
                con, query, limit=args.limit, detail=1, mode=mode, tier=args.tier,
                rerank=RERANK[args.rerank],
            )["results"]
            ids = [r["id"] for r in results]
            rank = ids.index(memory_id) + 1 if memory_id in ids else args.limit + 1
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

    if args.rerank != "off":
        con.close()
        return

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
