"""Can recall tell when it has nothing? Measure it before building anything on it.

`mem_recall` always returns its window: a question the store cannot answer still gets
five results, ranked, scored, and indistinguishable in shape from a real hit. Whether
any signal the pipeline already computes separates the two is an empirical question,
and this answers it without changing recall.

Answerable queries are the two authored fixtures (cold 72, hot 38). Unanswerable ones
are `tests/data/abstention_queries.json`, whose key terms were checked against every
memory before they were written down. Four signals, all about the top of the response:

- fused   -- the RRF score of the first result, as recall returns it
- lanes   -- how many lanes hold the first result in their candidate pool (the
             reranker gate's own test: 3 means it skips the reranker)
- cosine  -- best chunk similarity between the query and the first result: the one
             absolute number here, since RRF only knows ranks
- logit   -- the Korean cross-encoder's best logit over the top 10

For each, AUC says how well it ranks answerable above unanswerable (0.5 = coin), and
the table reads off what a cut-off would cost: to turn away a given share of the
unanswerable, what share of the answerable -- and of the ones whose answer really was
in the window -- would be turned away with them.

    python scripts/eval_abstention.py
    python scripts/eval_abstention.py --no-rerank     # skip the logit signal (fast)

Measure on a clean copy (`scripts/make_clean_copy.py`) with `AGENT_MEMORY_HOT_PATHS`
pointed at a missing directory, like every ranking measurement here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, embedding, rerank, store  # noqa: E402
from eval_paraphrase import FIXTURE, resolve, resolve_hot  # noqa: E402

ABSTENTION = FIXTURE.parent / "abstention_queries.json"
HOT = FIXTURE.parent / "hot_queries.json"
WINDOW = 5


def answerable(con) -> list[tuple[str, str]]:
    out = []
    for path in (FIXTURE, HOT):
        for case in json.loads(path.read_text())["cases"]:
            gold = (resolve_hot(con, case["path"], case["heading"]) if "path" in case
                    else resolve(con, case["topic_match"]))
            if gold is not None:
                out.append((case["query"], gold))
    return out


def signals(con, query: str, use_rerank: bool) -> tuple[dict, list[str]]:
    response = store.recall(con, query, limit=WINDOW, detail=3, rerank=False)
    results = response["results"]
    top = results[0]
    q = embedding.encode_query(query)
    cosine = max(
        float(embedding.from_blob(row[0]) @ q)
        for row in con.execute(
            "SELECT embedding FROM memory_chunks WHERE memory_id = ?", (top["id"],))
    )
    out = {"fused": top["score"], "lanes": len(top["lanes"]), "cosine": cosine}
    if use_rerank:
        # The head the production reranker scores is the fused order *before* tier
        # reservation and lane rescue reshape the window, so both are off here.
        slots, rescue = config.MIN_TIER_SLOTS, config.RESCUE_LANE_LEADERS
        config.MIN_TIER_SLOTS, config.RESCUE_LANE_LEADERS = 0, 0
        try:
            head = store.recall(con, query, limit=config.RERANK_DEPTH, detail=3,
                                rerank=False)["results"]
        finally:
            config.MIN_TIER_SLOTS, config.RESCUE_LANE_LEADERS = slots, rescue
        out["logit"] = max(rerank.scores(
            query, [f"{r['topic']}\n{r['content']}" for r in head]))
    return out, [r["id"] for r in results]


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random answerable scores above a random unanswerable), ties counted half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true", help="skip the logit signal")
    ap.add_argument("--dump", type=Path, help="write per-case signals as JSON")
    args = ap.parse_args()
    con = db.connect()

    rows = []
    for query, gold in answerable(con):
        s, ids = signals(con, query, not args.no_rerank)
        rows.append({"kind": "answerable", "hit": gold in ids, "query": query, **s})
    for case in json.loads(ABSTENTION.read_text())["cases"]:
        s, _ = signals(con, case["query"], not args.no_rerank)
        rows.append({"kind": case["group"], "hit": False, "query": case["query"], **s})
    if args.dump:
        args.dump.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    ans = [r for r in rows if r["kind"] == "answerable"]
    hit = [r for r in ans if r["hit"]]
    groups = {"all unanswerable": [r for r in rows if r["kind"] != "answerable"],
              "off_domain": [r for r in rows if r["kind"] == "off_domain"],
              "near_domain": [r for r in rows if r["kind"] == "near_domain"]}
    print(f"answerable {len(ans)} (answer in top {WINDOW}: {len(hit)}) · "
          + " · ".join(f"{k} {len(v)}" for k, v in groups.items()))

    names = ["fused", "lanes", "cosine"] + ([] if args.no_rerank else ["logit"])
    print("\nAUC (answerable above unanswerable; 0.5 = no signal)")
    print(f"{'signal':8}" + "".join(f"{k:>18}" for k in groups))
    for name in names:
        print(f"{name:8}" + "".join(
            f"{auc([r[name] for r in ans], [r[name] for r in g]):18.3f}"
            for g in groups.values()))

    print("\nCost of a cut-off: the lowest cut that turns away at least this share of the"
          " unanswerable -> share of unanswerable it actually turns away, and share of"
          " answerable (of answered-in-window) turned away with them")
    neg = groups["all unanswerable"]
    for name in names:
        values = sorted(r[name] for r in neg)
        line = []
        for share in (0.5, 0.8, 1.0):
            cut = values[min(len(values) - 1, int(np.ceil(share * len(values))) - 1)]
            caught = sum(v <= cut for v in values) / len(values)
            lost = sum(r[name] <= cut for r in ans) / len(ans)
            lost_hit = sum(r[name] <= cut for r in hit) / len(hit)
            line.append(f">={share:4.0%}: {caught:5.1%} -> {lost:5.1%} ({lost_hit:5.1%})")
        print(f"{name:8}  " + "   ".join(line))

    print("\nMedians")
    for name in names:
        med = lambda rs: float(np.median([r[name] for r in rs]))  # noqa: E731
        print(f"{name:8}  answerable {med(ans):8.3f}   "
              + "   ".join(f"{k} {med(v):8.3f}" for k, v in groups.items()))


if __name__ == "__main__":
    main()
