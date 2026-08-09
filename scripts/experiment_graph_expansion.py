"""Does the concept graph improve recall if mixed into `mem_recall`?

`mem_graph` is a separate tool today: the agent traverses it deliberately, and recall
never consults it. This asks whether it should — by adding one graph lane to the same
RRF fusion the lexical and semantic lanes already go through, changing nothing else.

Three ways the graph can produce candidates:

  E  entity lane   — match the query against entity names, return the memories citing
                     the matched entities. An independent signal: it does not depend on
                     the base lanes finding anything first.
  C  co-citation   — pseudo-relevance feedback. Take the base lanes' top-N, collect
                     their entities, return other memories sharing those entities.
  H  relation hop  — same seeds, but walk one relation edge first, so a memory can be
                     reached through a concept it shares with no one directly. This is
                     the "skip a memory to reach the third one" path the graph exists for.

⚠️ Only cold memories are annotated (106/106 cold, 0/13 hot), and every fixture target
is cold. A lane that can only ever emit cold memories therefore has a structural
advantage on this fixture that would not survive a hot target. Read `--tier cold`, where
that advantage is neutralised, as the honest number.

Read-only: builds nothing, writes nothing.

    python scripts/experiment_graph_expansion.py
    python scripts/experiment_graph_expansion.py --tier cold
"""

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, korean, store  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "paraphrase_queries.json"
DEPTH = 20
SEEDS = 3           # base-lane results used as feedback seeds for C and H
ENTITY_COVER = 0.5  # fraction of an entity's tokens that the query must contain


def resolve(con, prefix: str) -> str | None:
    rows = con.execute(
        "SELECT id FROM memories WHERE substr(topic, 1, ?) = ?", (len(prefix), prefix)
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


# ------------------------------------------------------------------ graph structures


def load_graph(con):
    """entity -> memories, memory -> entities, entity -> related entities."""
    ent_to_mem: dict[str, set[str]] = {}
    mem_to_ent: dict[str, set[str]] = {}
    for r in con.execute("SELECT memory_id, entity FROM memory_entities"):
        ent_to_mem.setdefault(r["entity"], set()).add(r["memory_id"])
        mem_to_ent.setdefault(r["memory_id"], set()).add(r["entity"])

    adjacent: dict[str, set[str]] = {}
    for r in con.execute("SELECT source, target FROM relations"):
        adjacent.setdefault(r["source"], set()).add(r["target"])
        adjacent.setdefault(r["target"], set()).add(r["source"])

    # Entity names are normalized join keys; tokenize the display form the same way
    # the lexical lane tokenizes text, so "패치" in a query meets "패치를" in a name.
    tokens = {}
    for r in con.execute("SELECT name, display FROM entities"):
        toks = set(korean.tokens(r["display"]))
        if toks:
            tokens[r["name"]] = toks
    return ent_to_mem, mem_to_ent, adjacent, tokens


def _rank_by_weight(weights: dict[str, float], pool: int) -> list[str]:
    # Tie-break on id, and accumulate in sorted order below: entity sets iterate in
    # hash order, which varies per process, and float addition is not associative --
    # without both, a rerun reorders ties and the marginal comparisons move by a case.
    return [m for m, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:pool]]


def _spread(seed_entities: set[str], ent_to_mem, exclude: set[str], pool: int) -> list[str]:
    """Memories citing the seed entities, rarer entities counting for more.

    `exclude` is normally empty: dropping the seeds is right for expansion (find what
    the base lanes missed) but wrong for a fused lane, because it removes the memory
    the base lanes already ranked first and hands its slot to a neighbour.
    """
    weights: dict[str, float] = {}
    for ent in sorted(seed_entities):
        mems = ent_to_mem.get(ent, ())
        if not mems:
            continue
        w = 1.0 / len(mems)  # an entity on 40 memories discriminates nothing
        for mid in mems:
            if mid not in exclude:
                weights[mid] = weights.get(mid, 0.0) + w
    return _rank_by_weight(weights, pool)


def lane_entity(query, ent_to_mem, tokens, pool) -> list[str]:
    qt = set(korean.tokens(query))
    if not qt:
        return []
    weights: dict[str, float] = {}
    for ent, toks in sorted(tokens.items()):
        covered = len(toks & qt) / len(toks)
        if covered < ENTITY_COVER:
            continue
        mems = ent_to_mem.get(ent, ())
        if not mems:
            continue
        w = covered / len(mems)
        for mid in mems:
            weights[mid] = weights.get(mid, 0.0) + w
    return _rank_by_weight(weights, pool)


def lane_cocitation(seeds, ent_to_mem, mem_to_ent, pool, drop_seeds=False) -> list[str]:
    seed_entities = set().union(*(mem_to_ent.get(s, set()) for s in seeds)) if seeds else set()
    return _spread(seed_entities, ent_to_mem, set(seeds) if drop_seeds else set(), pool)


def lane_hop(seeds, ent_to_mem, mem_to_ent, adjacent, pool) -> list[str]:
    direct = set().union(*(mem_to_ent.get(s, set()) for s in seeds)) if seeds else set()
    hopped: set[str] = set()
    for ent in direct:
        hopped |= adjacent.get(ent, set())
    # Only the newly reached concepts, else this collapses into co-citation.
    return _spread(hopped - direct, ent_to_mem, set(), pool)


# --------------------------------------------------------------------------- ranking


def order(con, lanes: dict[str, list[str]], tier: str | None,
          weights: dict[str, float] | None = None) -> list[str]:
    """Production ordering: RRF, then recency decay, then tier slot reservation.

    `weights` scales a lane's RRF contribution. Plain RRF treats every lane as equally
    trustworthy, which decides the outcome in advance when one lane is much noisier
    than the others -- so the graph lane gets to be tried quietly as well as loudly.
    """
    scores: dict[str, float] = {}
    for name, ids in lanes.items():
        w = (weights or {}).get(name, 1.0)
        for rank, mid in enumerate(ids, start=1):
            scores[mid] = scores.get(mid, 0.0) + w / (config.RRF_K + rank)
    if not scores:
        return []
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
        key=lambda kv: -kv[1],
    )
    if tier is None:
        ranked = store._reserve_tier_slots(ranked, rows, DEPTH, config.MIN_TIER_SLOTS)
    return [m for m, _ in ranked]


def margin(lanes) -> float:
    """top1 / top2 fused score. 1.0 means the base lanes could not separate them."""
    scores: dict[str, float] = {}
    for ids in lanes.values():
        for rank, mid in enumerate(ids, start=1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (config.RRF_K + rank)
    top = sorted(scores.values(), reverse=True)[:2]
    if len(top) < 2 or top[1] == 0:
        return float("inf")
    return top[0] / top[1]


def report(name, ranks, base=None):
    n = len(ranks)
    r1, r5 = sum(r == 1 for r in ranks) / n, sum(r <= 5 for r in ranks) / n
    mrr = sum(1 / r for r in ranks) / n
    line = (f"  {name:26} R@1={r1:6.1%}  R@5={r5:6.1%}  MRR={mrr:.3f}  "
            f"median={int(statistics.median(ranks))}")
    if base is not None:
        better = sum(v < b for v, b in zip(ranks, base))
        worse = sum(v > b for v, b in zip(ranks, base))
        line += f"   moved: +{better} / -{worse}"
    print(line)
    return mrr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("hot", "cold"), default=None)
    args = ap.parse_args()

    con = db.connect()
    cases = []
    for c in json.loads(FIXTURE.read_text())["cases"]:
        gold = resolve(con, c["topic_match"])
        if gold:
            cases.append((gold, c["query"]))

    ent_to_mem, mem_to_ent, adjacent, tokens = load_graph(con)
    annotated = len(mem_to_ent)
    total = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"{len(cases)} cases | graph: {len(tokens)} entities, "
          f"{sum(len(v) for v in adjacent.values()) // 2} edges, "
          f"{annotated}/{total} memories annotated | "
          f"tier={args.tier or 'both'}\n")

    where, params = store._filters(None, None, args.tier)
    pool = config.CANDIDATE_POOL

    # (label, graph lanes to add, weight, how many candidates the graph lane may emit)
    variants = [
        ("E  entity", ["E"], 1.0, pool),
        ("E  entity  w=0.5", ["E"], 0.5, pool),
        ("E  entity  w=0.5 top10", ["E"], 0.5, 10),
        ("C  co-citation", ["C"], 1.0, pool),
        ("C  co-citation w=0.5", ["C"], 0.5, pool),
        ("C  co-citation w=0.25 top10", ["C"], 0.25, 10),
        ("C  seeds dropped w=0.5", ["Cx"], 0.5, pool),
        ("H  relation hop w=0.5", ["H"], 0.5, pool),
        ("H  relation hop w=0.25 top10", ["H"], 0.25, 10),
        ("E+C  w=0.5", ["E", "C"], 0.5, pool),
        ("E+C+H  w=0.25 top10", ["E", "C", "H"], 0.25, 10),
    ]
    ranks: dict[str, list[int]] = {"baseline": []} | {v[0]: [] for v in variants}
    graph_only: dict[str, list[int]] = {"E alone": [], "C alone": [], "H alone": []}
    emitted = {"E": 0, "C": 0, "H": 0}

    for gold, query in cases:
        base = {
            "lexical": store._lexical(con, query, where, params, pool),
            "semantic": store._semantic(con, query, where, params, pool),
        }
        base_order = order(con, base, args.tier)
        seeds = base_order[:SEEDS]

        g = {
            "E": lane_entity(query, ent_to_mem, tokens, pool),
            "C": lane_cocitation(seeds, ent_to_mem, mem_to_ent, pool),
            "Cx": lane_cocitation(seeds, ent_to_mem, mem_to_ent, pool, drop_seeds=True),
            "H": lane_hop(seeds, ent_to_mem, mem_to_ent, adjacent, pool),
        }
        for k in emitted:
            if g[k]:
                emitted[k] += 1

        def rank_of(ids):
            return ids.index(gold) + 1 if gold in ids[:DEPTH] else DEPTH + 1

        ranks["baseline"].append(rank_of(base_order))
        for label, keys, weight, cap in variants:
            lanes = dict(base)
            weights = {}
            for k in keys:
                lanes[f"graph_{k}"] = g[k][:cap]
                weights[f"graph_{k}"] = weight
            ranks[label].append(rank_of(order(con, lanes, args.tier, weights)))
        for k, label in (("E", "E alone"), ("C", "C alone"), ("H", "H alone")):
            graph_only[label].append(rank_of(g[k]))

    print("fused with the base lanes:")
    base_ranks = ranks["baseline"]
    report("baseline", base_ranks)
    for label, *_ in variants:
        report(label, ranks[label], base_ranks)

    print("\ngraph lane alone (is it a retrieval signal at all?):")
    for label, r in graph_only.items():
        report(label, r)

    # The always-on variants trade rescued cases for damaged ones. If the graph only
    # helps where the base lanes are already weak, gate it on that weakness instead.
    print("\nconditional: consult the graph only when the base lanes disagree or barely separate")
    gates: list[tuple[str, Callable[[float, bool], bool]]] = [
        ("lanes disagree on top-1", lambda m, agree: not agree),
        ("margin < 1.10", lambda m, agree: m < 1.10),
        ("margin < 1.25", lambda m, agree: m < 1.25),
        ("disagree or margin < 1.10", lambda m, agree: not agree or m < 1.10),
    ]
    for label, gate in gates:
        for weight in (0.5, 1.0):
            ranks_c, fired = [], 0
            for gold, query in cases:
                base = {
                    "lexical": store._lexical(con, query, where, params, pool),
                    "semantic": store._semantic(con, query, where, params, pool),
                }
                base_order = order(con, base, args.tier)
                lex, sem = base["lexical"], base["semantic"]
                agree = bool(lex and sem and lex[0] == sem[0])
                if gate(margin(base), agree):
                    fired += 1
                    seeds = base_order[:SEEDS]
                    lanes = dict(base)
                    lanes["graph_C"] = lane_cocitation(seeds, ent_to_mem, mem_to_ent, pool)
                    ids = order(con, lanes, args.tier, {"graph_C": weight})
                else:
                    ids = base_order
                ranks_c.append(ids.index(gold) + 1 if gold in ids[:DEPTH] else DEPTH + 1)
            report(f"{label} w={weight} ({fired} fired)", ranks_c, base_ranks)

    n = len(cases)
    print(f"\nlane produced candidates: E {emitted['E']}/{n} · C {emitted['C']}/{n} "
          f"· H {emitted['H']}/{n}")
    con.close()


if __name__ == "__main__":
    main()
