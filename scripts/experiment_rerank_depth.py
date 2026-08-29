"""How deep should the reranker look, now that the pool no longer holds the answer?

`RERANK_DEPTH` is 10 because in August the answer was inside the top 10 for 71 of 72
cold cases -- there was nothing to reach for. Measured 2026-08-29 that ceiling is
81.9%, and depth 20 and 50 raise it to 87.5% and 93.1%. So the 2026-08-10 verdict
("do not raise depth") lost its premise and this re-measures it.

One reranker pass per query prices every depth. Scoring the base top `POOL` in base
order makes a depth-k run derivable: reorder the first k by logit, leave the tail --
which is exactly what `store._rerank` does. So 50 inferences answer for 10, 20 and 50,
and `--dump` makes every later question about this data free.

Both gate settings are reported, because with the gate in production (6e32a96) the
question is no longer "what does depth cost every query" but "what does it cost the
18% that reach the reranker".

    python scripts/experiment_rerank_depth.py --dump /tmp/depth.json
    python scripts/experiment_rerank_depth.py --load /tmp/depth.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, store  # noqa: E402
from eval_paraphrase import FIXTURE  # noqa: E402
from experiment_rerank import load, rerank_scores  # noqa: E402
from experiment_rerank_gate import HOT_FIXTURE, fixture_cases, middle_cases  # noqa: E402

POOL = 50  # candidates retrieved and scored per query
WINDOW = 20  # ranks past this are a miss, as every other harness here counts them
DEPTHS = (10, 20, 50)


def metrics(ranks: list[int]) -> str:
    n = len(ranks)
    return (f"R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
            f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
            f"MRR={sum(1/r for r in ranks)/n:.3f}")


def reordered(row: dict, depth: int) -> int:
    """Rank of the gold memory after reranking the first `depth` candidates.

    Capped at `WINDOW` + 1 like the base column. Without the cap a depth-50 arm scores
    a gold left at rank 30 as 1/30 while the base column scores the same case as 1/21,
    and the MRR difference then reads as the depth having hurt.
    """
    head = list(zip(row["logits"][:depth], row["ids"][:depth]))
    order = [mid for _, mid in sorted(head, key=lambda s: (-s[0], s[1]))]
    ids = order + row["ids"][depth:]
    rank = ids.index(row["gold"]) + 1 if row["gold"] in ids else WINDOW + 1
    return min(rank, WINDOW + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle", type=int, default=60)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--load", type=Path, default=None)
    args = ap.parse_args()

    con = db.connect()
    if args.load:
        rows = json.loads(args.load.read_text())
        per_inference = None
        print(f"loaded {len(rows)} cases from {args.load}\n")
    else:
        from agent_memory import config

        sources = [("cold", fixture_cases(con, FIXTURE)),
                   ("hot", fixture_cases(con, HOT_FIXTURE))]
        if args.middle:
            sources.append(("middle", middle_cases(con, args.middle)))
        counts = store.sessions_tier_counts(con)
        print(f"store: hot {counts.get('hot', 0)} : cold {counts.get('cold', 0)}, "
              f"pool={POOL}, window={WINDOW}")
        print("sources: " + ", ".join(f"{n} {len(c)}" for n, c in sources) + "\n")

        model = load((args.model or config.RERANK_MODEL_DIR).expanduser(),
                     args.max_tokens)
        rows, spent, inferences = [], 0.0, 0
        for name, cases in sources:
            for gold, query in cases:
                results = store.recall(con, query, limit=POOL, detail=3,
                                       mode="hybrid", rerank=False)["results"]
                if not results:
                    continue
                ids = [r["id"] for r in results]
                start = time.perf_counter()
                logits = rerank_scores(
                    model, query, [f"{r['topic']}\n{r['content']}" for r in results]
                )
                spent += time.perf_counter() - start
                inferences += len(results)
                rows.append({
                    "source": name,
                    "gold": gold,
                    "ids": ids,
                    "logits": logits,
                    # The production gate, so the two settings below are the real ones.
                    "gated": store._corroborated(_lanes_of(results[0]), ids[0]),
                    "base": ids.index(gold) + 1 if gold in ids else WINDOW + 1,
                })
        per_inference = spent / max(1, inferences)
        print(f"scored {inferences} pairs at {per_inference*1000:.0f} ms each\n")
        if args.dump:
            args.dump.write_text(json.dumps(rows, ensure_ascii=False))
            print(f"dumped {len(rows)} cases to {args.dump}\n")

    source_names = list(dict.fromkeys(r["source"] for r in rows))

    def by_source(pick) -> str:
        parts = []
        for name in source_names:
            group = [pick(r) for r in rows if r["source"] == name]
            n = max(1, len(group))
            parts.append(f"{name} {sum(x == 1 for x in group)/n:.1%}"
                         f"/{sum(x <= 5 for x in group)/n:.1%}")
        return "  ".join(parts)

    # Ranks past WINDOW count as a miss, which is how the fixtures are scored; the base
    # column therefore matches eval_paraphrase and eval_hot exactly.
    base = [min(r["base"], WINDOW + 1) for r in rows]
    rate_gated = sum(not r["gated"] for r in rows) / len(rows)
    print(f"  {'config':<26}{'reranked':>10}{'mean latency':>14}   pooled")
    print(f"  {'no rerank':<26}{0:>9.0%}{'~10 ms':>14}   {metrics(base)}"
          f"   {by_source(lambda r: min(r['base'], WINDOW + 1))}")
    for depth in DEPTHS:
        for gated in (True, False):
            def pick(r, depth=depth, gated=gated):
                if gated and r["gated"]:
                    return min(r["base"], WINDOW + 1)
                return reordered(r, depth)
            rate = rate_gated if gated else 1.0
            cost = ("?" if per_inference is None
                    else f"~{rate * depth * per_inference:.1f} s")
            label = f"depth {depth}" + (" + gate" if gated else "")
            print(f"  {label:<26}{rate:>9.0%}{cost:>14}   "
                  f"{metrics([pick(r) for r in rows])}   {by_source(pick)}")
    con.close()


def _lanes_of(result: dict) -> dict:
    """The lane membership of the top hit, in the shape `_corroborated` reads.

    Rebuilt from the response rather than re-running the lanes: `recall` does not hand
    its lane lists back, and the response's per-result `lanes` is the same membership
    the gate reads in production.
    """
    return {name: [result["id"]] for name in result["lanes"]}


if __name__ == "__main__":
    main()
