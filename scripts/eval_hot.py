"""Report the hot fixture by block, project and authoring style.

`eval_paraphrase.py` prints one number per mode, which is the right shape for the
72-case baseline where every case is the same kind. The hot fixture is deliberately
not uniform: it carries two blocks, six projects and two authoring styles, and the
whole point of the splits is that the aggregate hides them -- a `topic_echo` case
gets the 3x heading boost, so mixing it with `id_recall` inflates the headline.

    python scripts/eval_hot.py
    python scripts/eval_hot.py --rerank
    python scripts/eval_hot.py --show-misses

For the hot_over_cold block it also reports which tier actually won, because the
failure that matters there is not "rank 7" but "the archived copy outranked the file
that corrected it".
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, store  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "hot_queries.json"
DEPTH = 20


def resolve_hot(con, path: str, heading: str) -> str | None:
    """Fixture paths are `~`-relative; the store's are absolute. See eval_paraphrase."""
    rows = con.execute(
        "SELECT id FROM memories WHERE tier = 'hot' AND path = ? AND topic = ?",
        (str(Path(path).expanduser()), heading[:120]),
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def line(label: str, ranks: list[int], width: int = 22) -> str:
    n = len(ranks)
    if not n:
        return f"  {label:<{width}} —"
    return (f"  {label:<{width}} n={n:<3} "
            f"R@1={sum(r == 1 for r in ranks) / n:6.1%}  "
            f"R@5={sum(r <= 5 for r in ranks) / n:6.1%}  "
            f"MRR={sum(1 / r for r in ranks) / n:.3f}  "
            f"median={int(statistics.median(ranks))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--mode", default="hybrid", choices=("lexical", "semantic", "hybrid"))
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--tier", choices=("hot", "cold"), default=None)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    cases = json.loads(args.fixture.read_text())["cases"]
    con = db.connect()

    rows = []
    for case in cases:
        gold = resolve_hot(con, case["path"], case["heading"])
        if gold is None:
            print(f"  ! did not resolve: {case['heading'][:50]}")
            continue
        results = store.recall(con, case["query"], limit=DEPTH, detail=1,
                               mode=args.mode, tier=args.tier,
                               rerank=args.rerank or None)["results"]
        ids = [r["id"] for r in results]
        rank = ids.index(gold) + 1 if gold in ids else DEPTH + 1
        rows.append({**case, "rank": rank,
                     "top_tier": results[0]["tier"] if results else None,
                     "top_topic": results[0]["topic"] if results else ""})

    tag = f"mode={args.mode}" + (", rerank" if args.rerank else "")
    tag += f", tier={args.tier}" if args.tier else ", both tiers"
    print(f"{len(rows)} hot cases ({tag})\n")
    print(line("overall", [r["rank"] for r in rows]))

    for key, title in (("block", "block"), ("authored_from", "authoring"), ("project", "project")):
        print(f"\n  -- by {title}")
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["rank"])
        for name in sorted(groups, key=lambda k: -len(groups[k])):
            print(line(name, groups[name]))

    contested = [r for r in rows if r["block"] == "hot_over_cold"]
    if contested:
        print("\n  -- hot_over_cold: did the hot copy win the top slot?")
        for r in contested:
            mark = "ok " if r["rank"] == 1 else ("COLD" if r["top_tier"] == "cold" else "hot ")
            print(f"    {mark} rank {r['rank']:>3}  top={r['top_tier']:<4} {r['query'][:44]}")
            if r["rank"] != 1:
                print(f"           won by: {r['top_topic'][:70]}")

    if args.show_misses:
        print("\n  -- misses (rank > 5)")
        for r in sorted((r for r in rows if r["rank"] > 5), key=lambda r: -r["rank"]):
            print(f"    rank {r['rank']:>3}  [{r['project'][:20]}] {r['query'][:50]}")
            print(f"           won by: {r['top_topic'][:70]}")

    con.close()


if __name__ == "__main__":
    main()
