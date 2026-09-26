"""The rerank gate's cost curve: how much accuracy each second of reranking buys.

Measured 2026-09-13 and again 2026-09-26: forcing the reranker past the gate lifts R@1
by 20 points or more, and the gate stays only because the reranker costs seconds a
query while a gated cold recall answers in 0.06 s. Only the two ends were ever
measured. This fills in the middle: reranker depth below 10, and gates that skip less
or more often than the shipped one.

    three   -- shipped: skip when all three lanes hold the fused top hit in their pool
    two     -- skip when lexical and semantic both hold it (the graph lane ignored)
    strict  -- skip only when lexical and semantic each rank it *first*
    off     -- always rerank

Every variant runs the production `store.recall` at the caller's window (5), with the
gate swapped in and `RERANK_DEPTH` set, so tier slots and lane rescue act as shipped.
Cross-encoder scores are cached per (query, passage): a pair scores the same in every
variant, so the whole grid costs about one depth-10 pass per query. Latency is reported
as passages scored per query times the measured cost of one passage.

    python scripts/experiment_gate_cost.py              # cold 72 + hot 38 + verbatim 60
    python scripts/experiment_gate_cost.py --middle 0   # the two authored fixtures only

Measure on a clean copy with `AGENT_MEMORY_HOT_PATHS` pointed at a missing directory.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, rerank, store  # noqa: E402
from eval_paraphrase import FIXTURE  # noqa: E402
from experiment_rerank_gate import HOT_FIXTURE, fixture_cases, middle_cases  # noqa: E402

WINDOW = 5
DEPTHS = (3, 5, 10)


def three(lanes, mid):
    return all(mid in lanes.get(name, ()) for name in ("lexical", "semantic", "graph"))


def two(lanes, mid):
    return all(mid in lanes.get(name, ()) for name in ("lexical", "semantic"))


def strict(lanes, mid):
    return all(lanes.get(name, [None])[0] == mid for name in ("lexical", "semantic"))


GATES = {"three": three, "two": two, "strict": strict, "off": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle", type=int, default=60)
    args = ap.parse_args()
    con = db.connect()

    sources = [("cold", fixture_cases(con, FIXTURE)), ("hot", fixture_cases(con, HOT_FIXTURE))]
    if args.middle:
        sources.append(("verbatim", middle_cases(con, args.middle)))
    print("sources: " + ", ".join(f"{n} {len(c)}" for n, c in sources))

    cache: dict[tuple[str, str], float] = {}
    spent = [0.0, 0]  # seconds, passages actually scored
    asked = [0]       # passages requested by recall, cached or not
    score = rerank.scores

    def cached(query, passages, batch_size=16):
        asked[0] += len(passages)
        todo = [p for p in passages if (query, p) not in cache]
        if todo:
            start = time.perf_counter()
            for p, s in zip(todo, score(query, todo, batch_size)):
                cache[(query, p)] = s
            spent[0] += time.perf_counter() - start
            spent[1] += len(todo)
        return [cache[(query, p)] for p in passages]

    rerank.scores = cached
    shipped_gate, shipped_depth = store._corroborated, config.RERANK_DEPTH
    table = []
    try:
        # Deepest first so the shallower heads are all cache hits.
        for depth in sorted(DEPTHS, reverse=True):
            config.RERANK_DEPTH = depth
            for name, gate in GATES.items():
                if gate is not None:
                    store._corroborated = gate
                asked[0] = 0
                ranks = {s: [] for s, _ in sources}
                for source, cases in sources:
                    for gold, query in cases:
                        ids = [r["id"] for r in store.recall(
                            con, query, limit=WINDOW, detail=1,
                            rerank=None if gate is not None else True)["results"]]
                        ranks[source].append(ids.index(gold) + 1 if gold in ids else WINDOW + 1)
                n = sum(len(c) for _, c in sources)
                table.append((depth, name, asked[0] / n, ranks))
    finally:
        rerank.scores, store._corroborated = score, shipped_gate
        config.RERANK_DEPTH = shipped_depth

    per = spent[0] / max(1, spent[1])
    print(f"cross-encoder: {per*1000:.0f} ms per passage ({spent[1]} scored)\n")
    names = [s for s, _ in sources]
    print(f"{'depth':>5} {'gate':7} {'passages/q':>10} {'mean s/q':>9}   "
          + "   ".join(f"{s:>13}" for s in names) + "   pooled R@1/R@5")
    for depth, name, pq, ranks in table:
        cells = []
        for s in names:
            r = ranks[s]
            cells.append(f"{sum(x == 1 for x in r)/len(r):5.1%}/{sum(x <= 5 for x in r)/len(r):5.1%}")
        pooled = [x for s in names for x in ranks[s]]
        print(f"{depth:>5} {name:7} {pq:>10.2f} {pq*per:>9.2f}   " + "   ".join(cells)
              + f"   {sum(x == 1 for x in pooled)/len(pooled):5.1%}/"
                f"{sum(x <= 5 for x in pooled)/len(pooled):5.1%}")


if __name__ == "__main__":
    main()
