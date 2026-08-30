"""Where does candidate generation lose the answer? The reranker cannot fix any of it.

Measured 2026-08-30: the fused top 10 holds the cold answer 81.9% of the time and the
reranker scores 80.6%, so it is already at its ceiling, and widening the rerank window
buys nothing. Everything left is upstream of ranking -- a lane that never retrieved the
memory at all. This says which lane, and how far away it was.

The 2026-08-09 diagnosis of the same question concluded there was nothing to fix
(pool 50 -> 100 -> 200 moved no metric, df-pruning was worse), but the corpus has since
roughly doubled and the in-pool rate fell 15 points. That verdict's premise is gone,
which is the only reason to re-run it.

    python scripts/diagnose_candidates.py               # cold 72 and hot 38
    python scripts/diagnose_candidates.py --sweep       # and the pool sweep

Per-lane rank is the gold's position in that lane's own ordering, `-` when the lane
never returned it. `bm25` is how deep the gold sits among *every* document the lexical
query matches, which is the number that decides whether a bigger pool could reach it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, korean, store  # noqa: E402
from eval_paraphrase import FIXTURE, resolve, resolve_hot  # noqa: E402

HOT_FIXTURE = FIXTURE.parent / "hot_queries.json"
ALL = 10**6  # "no limit", for measuring how deep a lane buries the answer


def cases_of(con, fixture: Path) -> list[tuple[str, str]]:
    out = []
    for case in json.loads(fixture.read_text())["cases"]:
        gold = (resolve_hot(con, case["path"], case["heading"]) if "path" in case
                else resolve(con, case["topic_match"]))
        if gold is not None:
            out.append((gold, case["query"]))
    return out


def position(ids: list[str], gold: str) -> int | None:
    return ids.index(gold) + 1 if gold in ids else None


def lane_view(con, query: str, gold: str, pool: int) -> dict:
    """Where each lane puts the gold, with and without the pool cut."""
    lex_all = store._lexical(con, query, "", [], ALL)
    sem_all = store._semantic(con, query, "", [], ALL)
    lanes = {"lexical": lex_all[:pool], "semantic": sem_all[:pool]}
    seeds = [
        mid for mid, _ in sorted(
            store._fuse(lanes, config.RRF_K).items(), key=lambda kv: (-kv[1], kv[0])
        )[: config.GRAPH_LANE_SEEDS]
    ]
    graph = store._graph_lane(con, seeds, "", [], pool)
    return {
        "lex": position(lanes["lexical"], gold),
        "sem": position(lanes["semantic"], gold),
        "graph": position(graph, gold),
        "bm25": position(lex_all, gold),
        "bm25_total": len(lex_all),
        "sem_total": len(sem_all),
    }


def overlap(con, query: str, gold: str) -> str:
    """Query terms the gold's indexed text actually contains, with their df."""
    row = con.execute(
        "SELECT topic, content FROM memories WHERE id = ?", (gold,)
    ).fetchone()
    if row is None:
        return ""
    gold_terms = set(korean.tokens(f"{row['topic']}\n{row['content']}"))
    shared = [t for t in korean.tokens(query) if t in gold_terms]
    total = con.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    out = []
    for term in shared:
        try:
            df = con.execute(
                "SELECT COUNT(*) AS n FROM memories_fts WHERE memories_fts MATCH ?",
                (f'"{term}"',),
            ).fetchone()["n"]
        except Exception:
            df = -1
        out.append(f"{term}({df/total:.0%})")
    return " ".join(out) if out else "(공유 용어 없음)"


def diagnose(con, name: str, cases: list[tuple[str, str]]) -> None:
    pool = config.CANDIDATE_POOL
    print(f"\n=== {name}: {len(cases)} cases, pool={pool}")
    buckets = {"둘 다 놓침": 0, "렉시컬만 놓침": 0, "시맨틱만 놓침": 0, "둘 다 데려옴": 0}
    misses = []
    for gold, query in cases:
        results = store.recall(con, query, limit=20, detail=1, mode="hybrid",
                               rerank=False)["results"]
        fused = position([r["id"] for r in results], gold) or 21
        view = lane_view(con, query, gold, pool)
        if view["lex"] is None and view["sem"] is None:
            buckets["둘 다 놓침"] += 1
        elif view["lex"] is None:
            buckets["렉시컬만 놓침"] += 1
        elif view["sem"] is None:
            buckets["시맨틱만 놓침"] += 1
        else:
            buckets["둘 다 데려옴"] += 1
        if fused > 5:
            misses.append((fused, query, view, gold))

    print("  풀 안에 정답이 있었나 (전 케이스):")
    for key, n in buckets.items():
        print(f"    {key:<12} {n:>3} / {len(cases)} = {n/len(cases):5.1%}")

    print(f"\n  미스 {len(misses)}건 (융합 5위 밖) — 레인별 위치:")
    for fused, query, view, gold in sorted(misses, key=lambda m: -m[0]):
        def cell(v):
            return f"{v:>4}" if v is not None else "   -"
        print(f"    fused {fused:>3}  lex{cell(view['lex'])} sem{cell(view['sem'])}"
              f" graph{cell(view['graph'])}   bm25 "
              f"{view['bm25'] if view['bm25'] else '-'}/{view['bm25_total']}"
              f"   {query[:44]}")
        if view["lex"] is None or (view["bm25"] or 0) > pool:
            print(f"                {overlap(con, query, gold)}")


def sweep(con, sources: dict) -> None:
    print("\n=== 후보 풀 스윕 (2026-08-09 에는 전부 불변이었다)")
    original = config.CANDIDATE_POOL
    try:
        for pool in (50, 100, 200):
            config.CANDIDATE_POOL = pool
            line = [f"  pool {pool:>3}"]
            for name, cases in sources.items():
                ranks = []
                for gold, query in cases:
                    results = store.recall(con, query, limit=50, detail=1,
                                           mode="hybrid", rerank=False)["results"]
                    ranks.append(position([r["id"] for r in results], gold) or 51)
                n = len(ranks)
                line.append(
                    f"{name} R@1={sum(r == 1 for r in ranks)/n:5.1%} "
                    f"R@5={sum(r <= 5 for r in ranks)/n:5.1%} "
                    f"in10={sum(r <= 10 for r in ranks)/n:5.1%} "
                    f"in50={sum(r <= 50 for r in ranks)/n:5.1%}"
                )
            print("   ".join(line))
    finally:
        config.CANDIDATE_POOL = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    con = db.connect()
    sources = {"cold": cases_of(con, FIXTURE), "hot": cases_of(con, HOT_FIXTURE)}
    counts = store.sessions_tier_counts(con)
    print(f"store: hot {counts.get('hot', 0)} : cold {counts.get('cold', 0)}, "
          f"RRF_K={config.RRF_K}, graph w={config.GRAPH_LANE_WEIGHT}")
    for name, cases in sources.items():
        diagnose(con, name, cases)
    if args.sweep:
        sweep(con, sources)
    con.close()


if __name__ == "__main__":
    main()
