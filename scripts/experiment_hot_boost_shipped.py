"""The hot boost again, on the shipped path: does anything survive the reranker?

`experiment_hot_boost.py` (2026-08-21) re-implements fusion and so has no reranker, no
gate and no lane rescue -- all three ship now. Its one open question was recorded as
"measure with the reranker on and off": the reranker already lifts `hot_over_cold`, so
the boost's gain may be one it no longer needs to buy.

Here `store._fuse` is wrapped and everything downstream is production: the boost
multiplies a hot row's fused score by (1 + b) -- the `mul` shape, the only one the
August sweep left standing -- before the gate, the reranker, tier slots and rescue act
on it. Three reranker settings: off, auto (the gate, shipped) and on (forced).

⚠️ Provisional by construction. The re-open conditions in the August record are not
met (real-use hot drops: 4 of the ~20 needed), and both fixtures are the ones whose
scores define the trade-off, so a win here is not a licence to ship.

    python scripts/experiment_hot_boost_shipped.py

Measure on a clean copy with `AGENT_MEMORY_HOT_PATHS` pointed at a missing directory.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, rerank, store  # noqa: E402
from eval_paraphrase import FIXTURE, resolve, resolve_hot  # noqa: E402

HOT_FIXTURE = FIXTURE.parent / "hot_queries.json"
WINDOW = 5
BOOSTS = (0.0, 0.1, 0.25, 0.5)
RERANK = {"off": False, "auto": None, "on": True}


def cases(con) -> list[tuple[str, str, str]]:
    out = []
    for case in json.loads(FIXTURE.read_text())["cases"]:
        gold = resolve(con, case["topic_match"])
        if gold:
            out.append(("cold", gold, case["query"]))
    for case in json.loads(HOT_FIXTURE.read_text())["cases"]:
        gold = resolve_hot(con, case["path"], case["heading"])
        if gold:
            out.append((case["block"], gold, case["query"]))
    return out


def main() -> None:
    con = db.connect()
    tier = dict(con.execute("SELECT id, tier FROM memories").fetchall())
    rows = cases(con)
    groups = list(dict.fromkeys(g for g, _, _ in rows))
    print("cases: " + ", ".join(f"{g} {sum(r[0] == g for r in rows)}" for g in groups))

    fuse, score = store._fuse, rerank.scores
    cache: dict[tuple[str, str], float] = {}

    def cached(query, passages, batch_size=16):
        todo = [p for p in passages if (query, p) not in cache]
        for p, s in zip(todo, score(query, todo, batch_size) if todo else []):
            cache[(query, p)] = s
        return [cache[(query, p)] for p in passages]

    boost = [0.0]

    def boosted(lanes, k, weights=None):
        scores = fuse(lanes, k, weights)
        # recall fuses twice: once without weights to seed the graph lane, once with
        # them for the final order. Boost only the second, as the August sweep did, so
        # this measures the boost and not a change in which memories seed the graph.
        if boost[0] and weights is not None:
            scores = {m: s * (1 + boost[0]) if tier.get(m) == "hot" else s
                      for m, s in scores.items()}
        return scores

    rerank.scores, store._fuse = cached, boosted
    try:
        print(f"\n{'rerank':6} {'b':>5}   " + "   ".join(f"{g:>20}" for g in groups))
        for mode, flag in RERANK.items():
            for b in BOOSTS:
                boost[0] = b
                ranks = {g: [] for g in groups}
                for group, gold, query in rows:
                    ids = [r["id"] for r in store.recall(
                        con, query, limit=WINDOW, detail=1, rerank=flag)["results"]]
                    ranks[group].append(ids.index(gold) + 1 if gold in ids else WINDOW + 1)
                cells = []
                for g in groups:
                    r = ranks[g]
                    cells.append(f"{sum(x == 1 for x in r)/len(r):5.1%}/"
                                 f"{sum(x <= 5 for x in r)/len(r):5.1%}/"
                                 f"{sum(1/x for x in r)/len(r):.3f}")
                print(f"{mode:6} {b:>5}   " + "   ".join(f"{c:>20}" for c in cells))
    finally:
        rerank.scores, store._fuse = score, fuse


if __name__ == "__main__":
    main()
