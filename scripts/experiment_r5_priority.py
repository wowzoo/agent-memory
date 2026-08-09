"""R@5 first: re-measure the two deferred fusion settings side by side.

Two candidates were each measured on a different day and each bought R@5 or sold it:

  β  conviction weight — fused score becomes `Σ w/(k+r) + β·max w/(k+r)`, so a lane
     that is *sure* counts for more than two lanes that are lukewarm. Recorded as
     R@1 73.6→76.4% with R@5 95.8→94.4% (one case lost).
  C  graph co-citation lane at w=0.25, capped at 10 candidates. Recorded as R@5 +1..2
     cases with R@1 down 3.

They cannot be compared across those runs. `memories_fts` holds both tiers in one
table and `bm25()` derives IDF and average document length table-wide, so editing any
hot file reorders cold results -- a rebuild alone moved cold R@5 by a case. Hence one
process, one state, every variant, both tier modes.

The decision rule this script serves: R@5 wins, MRR breaks ties, R@1 is informational.
`RECALL_LIMIT` is 5, so R@5 is "did the answer reach the agent at all".

Read-only: builds nothing, writes nothing.

    python scripts/experiment_r5_priority.py
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, store  # noqa: E402
from experiment_graph_expansion import resolve  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "paraphrase_queries.json"
DEPTH = 20


def order(con, lanes: dict[str, list[str]], tier: str | None,
          weights: dict[str, float] | None = None, beta: float = 0.0) -> list[str]:
    """Production ordering with an optional conviction term.

    Plain RRF sums lane contributions, so corroboration is the only thing it rewards.
    `beta` adds the single best lane contribution back on top: at β=1 a memory that one
    lane ranks first is worth roughly twice what that lane alone says, which is what
    lets a semantic top-3 hit that BM25 never retrieved survive fusion.
    """
    summed: dict[str, float] = {}
    best: dict[str, float] = {}
    for name, ids in sorted(lanes.items()):
        w = (weights or {}).get(name, 1.0)
        for rank, mid in enumerate(ids, start=1):
            c = w / (config.RRF_K + rank)
            summed[mid] = summed.get(mid, 0.0) + c
            if c > best.get(mid, 0.0):
                best[mid] = c
    if not summed:
        return []
    scores = {m: s + beta * best[m] for m, s in summed.items()}
    rows = {
        r["id"]: r
        for r in con.execute(
            "SELECT id, created_at, tier FROM memories "
            f"WHERE id IN ({','.join('?' * len(scores))})",
            list(scores),
        )
    }
    ranked = sorted(
        ((m, s * store._recency(rows[m]["created_at"], config.HALF_LIFE_DAYS))
         for m, s in scores.items() if m in rows),
        # Tie-break on id: float accumulation order and set iteration are not stable
        # across processes, and this experiment argues over single cases.
        key=lambda kv: (-kv[1], kv[0]),
    )
    if tier is None:
        ranked = store._reserve_tier_slots(ranked, rows, DEPTH, config.MIN_TIER_SLOTS)
    return [m for m, _ in ranked]


def report(name, ranks, base=None):
    n = len(ranks)
    r1 = sum(r == 1 for r in ranks)
    r5 = sum(r <= 5 for r in ranks)
    mrr = sum(1 / r for r in ranks) / n
    line = (f"  {name:24} R@1={r1/n:6.1%} ({r1:2})  R@5={r5/n:6.1%} ({r5:2})  "
            f"MRR={mrr:.3f}  median={int(statistics.median(ranks))}")
    if base is not None:
        gained = sum(1 for v, b in zip(ranks, base) if v <= 5 < b)
        lost = sum(1 for v, b in zip(ranks, base) if b <= 5 < v)
        line += (f"   ΔR@5={r5 - sum(r <= 5 for r in base):+d}"
                 f" (in {gained} / out {lost})")
    print(line)


def run(con, cases, tier) -> None:
    where, params = store._filters(None, None, tier)
    pool = config.CANDIDATE_POOL

    # (label, beta, graph lane weight, graph candidate cap)
    variants = [
        ("baseline", 0.0, None, 0),
        ("β=0.25", 0.25, None, 0),
        ("β=0.5", 0.5, None, 0),
        ("β=1.0", 1.0, None, 0),
        ("β=2.0", 2.0, None, 0),
        ("C w=0.25 top10", 0.0, 0.25, 10),
        ("C w=0.25", 0.0, 0.25, pool),
        ("C w=0.35", 0.0, 0.35, pool),
        ("C w=0.5", 0.0, 0.5, pool),
        ("C w=0.65", 0.0, 0.65, pool),
        ("C w=0.75", 0.0, 0.75, pool),
        ("C w=1.0", 0.0, 1.0, pool),
        ("C w=0.25 top10 + β=1.0", 1.0, 0.25, 10),
        ("C w=0.5 + β=1.0", 1.0, 0.5, pool),
    ]
    ranks: dict[str, list[int]] = {v[0]: [] for v in variants}

    for gold, query in cases:
        base = {
            "lexical": store._lexical(con, query, where, params, pool),
            "semantic": store._semantic(con, query, where, params, pool),
        }
        # Seeds and candidates both come from the production code path, so a variant
        # here cannot mean something different from the same setting in `recall()`.
        seeds = [
            mid for mid, _ in sorted(
                store._fuse(base, config.RRF_K).items(), key=lambda kv: (-kv[1], kv[0])
            )[: config.GRAPH_LANE_SEEDS]
        ]
        cocit = store._graph_lane(con, seeds, where, params, pool)
        for label, beta, gw, cap in variants:
            lanes, weights = dict(base), {}
            if gw is not None:
                lanes["graph_C"] = cocit[:cap]
                weights["graph_C"] = gw
            ids = order(con, lanes, tier, weights, beta)
            ranks[label].append(ids.index(gold) + 1 if gold in ids[:DEPTH] else DEPTH + 1)

    print(f"tier={tier or 'both'}")
    base_ranks = ranks["baseline"]
    report("baseline", base_ranks)
    for label, *_ in variants[1:]:
        report(label, ranks[label], base_ranks)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.parse_args()

    con = db.connect()
    cases = []
    for c in json.loads(FIXTURE.read_text())["cases"]:
        gold = resolve(con, c["topic_match"])
        if gold:
            cases.append((gold, c["query"]))

    annotated = con.execute("SELECT COUNT(DISTINCT memory_id) FROM memory_entities").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"{len(cases)} cases | {annotated}/{total} memories annotated | "
          f"RRF_K={config.RRF_K} pool={config.CANDIDATE_POOL}\n")

    for tier in ("cold", None):
        run(con, cases, tier)
    con.close()


if __name__ == "__main__":
    main()
