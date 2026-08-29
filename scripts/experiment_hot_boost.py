"""Re-run the rejected hot boost, this time with hot gold on the other side of the scale.

`config.py` records the boost as tried and dropped: a flat score multiplier for hot
"dropped the fixture's cold targets to hybrid R@1 0% / R@5 40% (from 55% / 90%
cold-only), because it promoted files even for questions the files do not answer".

That judgement is worth re-opening for two reasons, and the second is the stronger one:

  1. the corpus grew -- hot was a few dozen sections in my own project then and is
     1070 now, against 206 archived memories.
  2. it was measured with 100% cold gold. A boost for hot lowers a cold-gold score by
     definition, so the experiment could only ever come back negative. That is the
     mirror image of the circularity already recorded for MIN_TIER_SLOTS ("정답이 100%
     cold 이므로 cold 슬롯을 늘리면 정의상 점수가 오른다 -- 순환이다"). With
     `hot_queries.json` there is finally a fixture the boost can *win* on, so the
     question becomes a trade-off curve instead of a one-sided loss.

Three shapes, because the recorded failure is explainable by shape rather than by idea:

  mul   fused * (1 + b) for hot rows. The faithful redo -- a flat multiplier lifts a
        pool-tail hot row by the same factor as a head one, which is exactly the
        "promoted files even for questions the files do not answer" failure.
  add   fused + b for hot rows. Scale-dependent: the RRF head sits near 0.09 per lane
        with k=10, so b is in hundredths.
  rank  inside each lane, a hot row at rank r contributes as if it ranked r - d. This
        is the shape the recorded failure argues for: 1/(k+r) is convex, so a rank
        shift lifts head rows much more than tail rows, and a hot row sitting at 40
        barely moves.

Method follows experiment_r5_priority.py: one process, one state, every variant, both
fixtures. Lanes are computed once per query and every variant re-scores the same lane
output, so the sweep costs one recall per case rather than one per case per variant.
Graph-lane seeds stay unboosted in all variants -- in production the boost would land
in `final_score`, after the graph lane has already been seeded -- so the sweep measures
the boost and not a change in seeding.

Decision rule, same as the R@5-first experiment: R@5 wins, MRR breaks ties, R@1 is
informational. `RECALL_LIMIT` is 5, so both limits are reported: 20 to stay comparable
with the recorded fixture numbers, 5 because that is what an agent actually receives
(and `MIN_TIER_SLOTS` only touches the tail of the window, so it bites at 5, not 20).

Read-only: builds nothing, writes nothing.

    python scripts/experiment_hot_boost.py
    python scripts/experiment_hot_boost.py --limit 5
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLD_FIXTURE = ROOT / "tests" / "data" / "paraphrase_queries.json"
HOT_FIXTURE = ROOT / "tests" / "data" / "hot_queries.json"
DEPTH = 20

VARIANTS = [
    ("baseline", None, 0.0),
    ("mul", "mul", 0.02),
    ("mul", "mul", 0.05),
    ("mul", "mul", 0.10),
    ("mul", "mul", 0.25),
    ("mul", "mul", 0.50),
    ("add", "add", 0.002),
    ("add", "add", 0.005),
    ("add", "add", 0.010),
    ("add", "add", 0.020),
    ("mul", "mul", 1.00),
    ("mul", "mul", 2.00),
    ("mul", "mul", 5.00),
    ("rank", "rank", 1),
    ("rank", "rank", 2),
    ("rank", "rank", 3),
    ("rank", "rank", 5),
    # The limit of a monotone lift: hot strictly above cold, relevance only breaking
    # ties inside a tier. Brackets the sweep -- "strict tier interleaving" was rejected
    # too, so this is the other rejected design's neighbour.
    ("tier-first", "tierfirst", 0.0),
]


def resolve(con, case) -> str | None:
    if "path" in case:
        rows = con.execute(
            "SELECT id FROM memories WHERE tier = 'hot' AND path = ? AND topic = ?",
            (case["path"], case["heading"][:120]),
        ).fetchall()
    else:
        prefix = case["topic_match"]
        rows = con.execute(
            "SELECT id FROM memories WHERE substr(topic, 1, ?) = ?",
            (len(prefix), prefix),
        ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def lanes_for(con, query: str) -> tuple[dict, dict]:
    """The hybrid lane output for one query, plus the row metadata for every candidate."""
    pool = config.CANDIDATE_POOL
    lanes = {
        "lexical": store._lexical(con, query, "", [], pool),
        "semantic": store._semantic(con, query, "", [], pool),
    }
    if config.GRAPH_LANE_WEIGHT > 0:
        base = store._fuse(lanes, config.RRF_K)
        seeds = [
            mid for mid, _ in sorted(base.items(), key=lambda kv: (-kv[1], kv[0]))
        ][: config.GRAPH_LANE_SEEDS]
        graph_ids = store._graph_lane(con, seeds, "", [], pool)
        if graph_ids:
            lanes["graph"] = graph_ids
    ids = {mid for lane in lanes.values() for mid in lane}
    rows = {}
    if ids:
        rows = {
            r["id"]: r
            for r in con.execute(
                "SELECT id, tier, topic FROM memories "
                f"WHERE id IN ({','.join('?' * len(ids))})",
                list(ids),
            )
        }
    return lanes, rows


def scores_for(lanes: dict, rows: dict, shape: str | None, b: float) -> dict:
    weights = {"graph": config.GRAPH_LANE_WEIGHT} if "graph" in lanes else {}
    k = config.RRF_K

    if shape == "rank":
        scores: dict[str, float] = {}
        for name, ids in sorted(lanes.items()):
            w = weights.get(name, 1.0)
            for rank, mid in enumerate(ids, start=1):
                hot = rows.get(mid) is not None and rows[mid]["tier"] == "hot"
                r = max(1, rank - int(b)) if hot else rank
                scores[mid] = scores.get(mid, 0.0) + w / (k + r)
        return scores

    scores = store._fuse(lanes, k, weights)
    if not shape:
        return scores
    # A constant larger than any fused score, so adding it puts every hot row above
    # every cold one while relevance still orders within each tier.
    ceiling = (max(scores.values()) + 1.0) if scores else 1.0
    for mid in scores:
        row = rows.get(mid)
        if row is None or row["tier"] != "hot":
            continue
        if shape == "mul":
            scores[mid] *= 1 + b
        elif shape == "add":
            scores[mid] += b
        else:  # tierfirst
            scores[mid] += ceiling
    return scores


def rank_of(scores: dict, rows: dict, gold: str, limit: int) -> int:
    ranked = sorted(
        (kv for kv in scores.items() if kv[0] in rows),
        key=lambda kv: (-kv[1], kv[0]),
    )
    ordered = store._reserve_tier_slots(ranked, rows, limit, config.MIN_TIER_SLOTS)
    ids = [mid for mid, _ in ordered[:DEPTH]]
    return ids.index(gold) + 1 if gold in ids else DEPTH + 1


def stats(ranks: list[int]) -> tuple[float, float, float]:
    n = len(ranks) or 1
    return (
        sum(r == 1 for r in ranks) / n,
        sum(r <= 5 for r in ranks) / n,
        sum(1 / r for r in ranks) / n,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="recall window the tier slots act on; default runs 20 and 5")
    args = ap.parse_args()
    limits = [args.limit] if args.limit else [DEPTH, config.RECALL_LIMIT]

    con = db.connect()
    prepared = {}
    for name, path in (("cold 72", COLD_FIXTURE), ("hot 38", HOT_FIXTURE)):
        cases = json.loads(path.read_text())["cases"]
        items = []
        for case in cases:
            gold = resolve(con, case)
            if gold is None:
                print(f"  ! did not resolve: {(case.get('topic_match') or case['heading'])[:46]}")
                continue
            lanes, rows = lanes_for(con, case["query"])
            items.append({"gold": gold, "lanes": lanes, "rows": rows,
                          "block": case.get("block", "-")})
        prepared[name] = items
        print(f"{name}: {len(items)} cases prepared")

    for limit in limits:
        print(f"\n{'=' * 96}\nlimit={limit}  (tier slots reserve {config.MIN_TIER_SLOTS})")
        head = (f"{'variant':>14} | {'cold 72 (R@1/R@5/MRR)':>26} | {'hot_only 26':>26} "
                f"| {'hot_over_cold 12':>26}")
        print(head)
        print("-" * len(head))
        for label, shape, b in VARIANTS:
            groups = {"cold 72": [], "hot_only": [], "hot_over_cold": []}
            for name in ("cold 72", "hot 38"):
                for it in prepared[name]:
                    sc = scores_for(it["lanes"], it["rows"], shape, b)
                    r = rank_of(sc, it["rows"], it["gold"], limit)
                    groups["cold 72" if name == "cold 72" else it["block"]].append(r)
            cells = []
            for key in ("cold 72", "hot_only", "hot_over_cold"):
                r1, r5, mrr = stats(groups[key])
                cells.append(f"{r1:6.1%} {r5:6.1%}  {mrr:.3f}")
            tag = label if shape is None else f"{label} {b}"
            print(f"{tag:>14} | {cells[0]:>26} | {cells[1]:>26} | {cells[2]:>26}")

    con.close()


if __name__ == "__main__":
    main()
