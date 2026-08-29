"""Should the reranker run at all? Ask the base ranking first.

The reranker is off by default because it costs 3 s, so ordinary recalls never get the
19 points it buys. But it does not buy points everywhere: on queries lifted verbatim
from a memory's middle it *destroys* the base order (measured 2026-08-29: R@1
81.7% -> 33.3%), and those are exactly the queries where the base lanes were already
confident. So the question is not "is 3 s worth it" but "can the base ranking tell us
in advance which shape this query is".

Every signal here is computed from the un-reranked response alone, because a gate that
needs the reranker's output cannot save the reranker's cost. Both orders are measured
per case and the sweep is offline, so one pass prices every threshold.

    python scripts/experiment_rerank_gate.py                # cold 72 + hot 38 + middle 60
    python scripts/experiment_rerank_gate.py --middle 0     # the two authored fixtures only
    python scripts/experiment_rerank_gate.py --dump /tmp/gate.json   # measure once
    python scripts/experiment_rerank_gate.py --load /tmp/gate.json   # re-sweep for free

`always base` and `always rerank` are the bounds a gate has to beat; `oracle` picks the
better order per case and is unreachable -- it is there to show how much a perfect gate
would be worth before anyone tunes a real one.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, store  # noqa: E402
from eval_paraphrase import FIXTURE, resolve, resolve_hot  # noqa: E402
from eval_recall import middle_sentence  # noqa: E402

HOT_FIXTURE = FIXTURE.parent / "hot_queries.json"
DEPTH = 20


def fixture_cases(con, path: Path) -> list[tuple[str, str]]:
    out = []
    for case in json.loads(path.read_text())["cases"]:
        gold = (resolve_hot(con, case["path"], case["heading"]) if "path" in case
                else resolve(con, case["topic_match"]))
        if gold is not None:
            out.append((gold, case["query"]))
    return out


def middle_cases(con, n: int) -> list[tuple[str, str]]:
    rows = con.execute(
        "SELECT id, content FROM memories WHERE tier = 'cold' ORDER BY id"
    ).fetchall()
    return [(r["id"], q) for r in rows if (q := middle_sentence(r["content"]))][:n]


def rank_of(gold: str, results: list) -> int:
    ids = [r["id"] for r in results]
    return ids.index(gold) + 1 if gold in ids else DEPTH + 1


def metrics(ranks: list[int]) -> str:
    n = len(ranks)
    return (f"R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
            f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
            f"MRR={sum(1/r for r in ranks)/n:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle", type=int, default=60,
                    help="how many middle-sentence cases to add (0 to skip)")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write the per-case ranks and signals here")
    ap.add_argument("--load", type=Path, default=None,
                    help="re-sweep a --dump file instead of measuring; costs no "
                         "reranker passes, which is the whole point of dumping")
    args = ap.parse_args()

    con = db.connect()
    if args.load:
        rows = json.loads(args.load.read_text())
        print(f"loaded {len(rows)} cases from {args.load}\n")
    else:
        sources = [("cold", fixture_cases(con, FIXTURE)),
                   ("hot", fixture_cases(con, HOT_FIXTURE))]
        if args.middle:
            sources.append(("middle", middle_cases(con, args.middle)))

        counts = store.sessions_tier_counts(con)
        print(f"store: hot {counts.get('hot', 0)} : cold {counts.get('cold', 0)}, "
              f"depth={DEPTH}")
        print("sources: " + ", ".join(f"{name} {len(c)}" for name, c in sources) + "\n")

        rows = []
        for name, cases in sources:
            for gold, query in cases:
                base = store.recall(con, query, limit=DEPTH, detail=1, mode="hybrid",
                                    rerank=False)["results"]
                rr = store.recall(con, query, limit=DEPTH, detail=1, mode="hybrid",
                                  rerank=True)["results"]
                if not base:
                    continue
                top, second = base[0], (base[1] if len(base) > 1 else None)
                rows.append({
                    "source": name,
                    "base": rank_of(gold, base),
                    "rerank": rank_of(gold, rr),
                    # Signals, all from the un-reranked response.
                    "margin": top["score"] - (second["score"] if second else 0.0),
                    "ratio": (second["score"] / top["score"]) if second and top["score"] else 0.0,
                    "lanes": len(top["lanes"]),
                    "top1": top["score"],
                })
        if args.dump:
            args.dump.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            print(f"dumped {len(rows)} cases to {args.dump}\n")

    # Order-preserving, so a loaded dump reports its columns in the order measured.
    source_names = list(dict.fromkeys(x["source"] for x in rows))

    def by_source(pick) -> str:
        """R@1/R@5 per source. Both, because 'no source gets worse' is the claim a
        gate has to support and R@1 alone cannot carry it."""
        parts = []
        for name in source_names:
            group = [pick(x) for x in rows if x["source"] == name]
            n = max(1, len(group))
            parts.append(f"{name} {sum(r == 1 for r in group)/n:.1%}"
                         f"/{sum(r <= 5 for r in group)/n:.1%}")
        return "  ".join(parts)

    print(f"  {'strategy':<28}{'rerank rate':>12}   pooled")
    for label, pick, rate in (
        ("always base", lambda x: x["base"], 0.0),
        ("always rerank", lambda x: x["rerank"], 1.0),
        # Unreachable by construction: it reads both orders before choosing, which
        # means paying for the reranker on every query it then discards.
        ("oracle (unreachable)", lambda x: min(x["base"], x["rerank"]), 1.0),
    ):
        ranks = [pick(x) for x in rows]
        print(f"  {label:<28}{rate:>11.0%}   {metrics(ranks)}"
              f"   R@1/R@5 by source: {by_source(pick)}")

    # A gate fires the reranker only when the base ranking looks *unconfident*. Each
    # signal is swept over its own deciles so the grid is data-driven rather than a
    # guess at the scale of an RRF score.
    for signal, confident_when_high in (("margin", True), ("ratio", False),
                                        ("top1", True), ("lanes", True)):
        values = sorted({x[signal] for x in rows})
        if len(values) > 11:
            qs = [statistics.quantiles(values, n=10)[i] for i in range(9)]
        else:
            qs = values
        print(f"\n  -- gate: rerank unless {signal} "
              f"{'>=' if confident_when_high else '<='} threshold")
        for tau in qs:
            def gated(x, tau=tau):
                confident = x[signal] >= tau if confident_when_high else x[signal] <= tau
                return x["base"] if confident else x["rerank"]
            ranks = [gated(x) for x in rows]
            rate = sum(not (x[signal] >= tau if confident_when_high else x[signal] <= tau)
                       for x in rows) / len(rows)
            print(f"    {signal} {tau:<10.4f}  rerank {rate:>5.0%}   {metrics(ranks)}"
                  f"   R@1/R@5 by source: {by_source(gated)}")

    con.close()


if __name__ == "__main__":
    main()
