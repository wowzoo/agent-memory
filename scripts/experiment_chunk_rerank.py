"""What should the cross-encoder read: the memory's opening, or its best chunk?

Production feeds `topic + body` truncated to RERANK_MAX_TOKENS, so evidence past that
window is invisible -- measured on `eval_recall.py`, whose queries are lifted from the
middle of a memory, R@5 falls 99.2% -> 79.5%. `rerank.py:22` names the fix (score every
chunk, keep the maximum) and prices it at four to five inferences per candidate.

This measures a third arm that costs nothing extra. `store._semantic` already computes
a cosine for *every* chunk in scope and throws all but the maximum away, so the chunk
that best answers the query is known before the reranker runs. Feeding that one chunk
keeps the inference count at one per candidate, and a shorter input is faster, not
slower. Whether a bi-encoder picks the chunk a cross-encoder would have picked is the
open question -- hence the maxpool arm, which is the ceiling for both.

    python scripts/experiment_chunk_rerank.py                                # cold 72
    python scripts/experiment_chunk_rerank.py --fixture tests/data/hot_queries.json
    python scripts/experiment_chunk_rerank.py --middle-sentence 100          # the shape
                                                     # the window is known to lose on

Read-only: no production path changes, and the arms share one retrieval pass so they
cannot disagree about what the base lanes returned.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, embedding, store  # noqa: E402
from eval_paraphrase import FIXTURE, resolve, resolve_hot  # noqa: E402
from eval_recall import middle_sentence  # noqa: E402
from experiment_rerank import load, rerank_scores  # noqa: E402

DEPTH = 20  # retrieval depth, as both eval harnesses use, so MRR stays comparable


def body_chunks(chunks: list[tuple[int, str, bytes]], topic: str) -> list:
    """The chunks holding prose.

    `embedding.passages` puts the topic in front of the body chunks when a memory has
    more than one, and a passage of `topic + topic` tells the cross-encoder nothing it
    does not already get from the prefix. Hot topics are stored truncated to 120 chars
    while the chunk carries the full heading, so the comparison runs both ways.
    """
    if len(chunks) > 1:
        first, head = chunks[0][1].strip(), topic.strip()
        if head and (first.startswith(head) or head.startswith(first)):
            return chunks[1:]
    return chunks


def report(label: str, ranks: list[int], base: list[int]) -> None:
    n = len(ranks)
    better = sum(r < b for r, b in zip(ranks, base))
    worse = sum(r > b for r, b in zip(ranks, base))
    print(f"  {label:<10} R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
          f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
          f"MRR={sum(1/r for r in ranks)/n:.3f}  "
          f"median={int(statistics.median(ranks))}  "
          f"(vs base: +{better} -{worse})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--model", type=Path, default=config.RERANK_MODEL_DIR)
    ap.add_argument("--depth", type=int, default=config.RERANK_DEPTH)
    ap.add_argument("--max-tokens", type=int, default=config.RERANK_MAX_TOKENS)
    ap.add_argument("--tier", choices=("hot", "cold"), default=None)
    ap.add_argument("--show-flips", action="store_true",
                    help="cases where an arm and the head arm disagree about rank")
    ap.add_argument("--middle-sentence", type=int, default=0, metavar="N",
                    help="ignore --fixture and use N eval_recall.py cases instead: a "
                         "sentence from the middle of a cold memory, which is the query "
                         "shape the 512-token window is known to lose on")
    args = ap.parse_args()

    con = db.connect()
    if args.middle_sentence:
        rows = con.execute(
            "SELECT id, content FROM memories WHERE tier = 'cold' ORDER BY id"
        ).fetchall()
        resolved = [
            (row["id"], query)
            for row in rows
            if (query := middle_sentence(row["content"]))
        ][: args.middle_sentence]
        source = f"middle sentences of {args.middle_sentence} cold memories"
    else:
        resolved = []
        for case in json.loads(args.fixture.read_text())["cases"]:
            gold = (resolve_hot(con, case["path"], case["heading"]) if "path" in case
                    else resolve(con, case["topic_match"]))
            if gold is None:
                key = case.get("topic_match") or case.get("heading", "")
                print(f"  ! did not resolve uniquely — skipped: {key[:50]}")
                continue
            resolved.append((gold, case["query"]))
        source = args.fixture.name

    counts = store.sessions_tier_counts(con)
    print(f"{len(resolved)} cases from {source}, depth={args.depth}, "
          f"window={args.max_tokens}, tier={args.tier or 'both'}")
    print(f"store: hot {counts.get('hot', 0)} : cold {counts.get('cold', 0)}\n")

    # One retrieval pass with the reranker off. Every arm reorders this same head.
    pools = []
    for gold, query in resolved:
        results = store.recall(con, query, limit=DEPTH, detail=3, mode="hybrid",
                               tier=args.tier, rerank=False)["results"]
        pools.append((gold, query, results))

    base_ranks = [
        ([r["id"] for r in results].index(gold) + 1
         if gold in [r["id"] for r in results] else DEPTH + 1)
        for gold, _, results in pools
    ]

    from tokenizers import Tokenizer

    model = load(args.model.expanduser(), args.max_tokens)
    # Its own instance: truncation is a property of the Tokenizer object, so counting
    # with the model's would report the truncated length and always look clean.
    counter = Tokenizer.from_file(str(args.model.expanduser() / "tokenizer.json"))
    counter.no_truncation()
    counter.no_padding()

    arms = ("head", "best", "maxpool")
    ranks: dict[str, list[int]] = {a: [] for a in arms}
    elapsed: dict[str, list[float]] = {a: [] for a in arms}
    inferences: dict[str, int] = {a: 0 for a in arms}
    cut = total = 0

    for gold, query, results in pools:
        head = results[: args.depth]
        chunks = {
            r["id"]: body_chunks(
                [(c["idx"], c["text"], c["embedding"]) for c in con.execute(
                    "SELECT idx, text, embedding FROM memory_chunks "
                    "WHERE memory_id = ? ORDER BY idx", (r["id"],))],
                r["topic"],
            )
            for r in head
        }
        qvec = embedding.encode_query(query)

        passages: dict[str, list[str]] = {a: [] for a in arms}
        spans: list[tuple[int, int]] = []  # maxpool: this candidate's slice
        for r in head:
            whole = f"{r['topic']}\n{r['content']}"
            passages["head"].append(whole)
            total += 1
            if len(counter.encode(whole).ids) > args.max_tokens:
                cut += 1

            texts = [text for _, text, _ in chunks[r["id"]]] or [r["content"]]
            sims = [
                float(embedding.from_blob(blob) @ qvec)
                for _, _, blob in chunks[r["id"]]
            ] or [0.0]
            passages["best"].append(f"{r['topic']}\n{texts[sims.index(max(sims))]}")
            start = len(passages["maxpool"])
            passages["maxpool"].extend(f"{r['topic']}\n{t}" for t in texts)
            spans.append((start, len(passages["maxpool"])))

        for arm in arms:
            begin = time.perf_counter()
            logits = rerank_scores(model, query, passages[arm])
            elapsed[arm].append(time.perf_counter() - begin)
            inferences[arm] += len(passages[arm])
            if arm == "maxpool":
                logits = [max(logits[a:b]) for a, b in spans]
            # Tie-break on id, as store._rerank does.
            order = sorted(zip(logits, [r["id"] for r in head]),
                           key=lambda s: (-s[0], s[1]))
            ids = [mid for _, mid in order] + [r["id"] for r in results[args.depth:]]
            ranks[arm].append(ids.index(gold) + 1 if gold in ids else DEPTH + 1)

    print(f"  truncated today: {cut}/{total} candidate passages exceed "
          f"{args.max_tokens} tokens = {cut/total:.1%}\n")
    report("base", base_ranks, base_ranks)
    for arm in arms:
        report(arm, ranks[arm], base_ranks)
    print()
    n = len(resolved)
    for arm in arms:
        print(f"  {arm:<10} {statistics.mean(elapsed[arm])*1000:6.0f} ms/query mean, "
              f"{inferences[arm]/n:5.2f} inferences/candidate-set-pass "
              f"({inferences[arm]} total)")

    for arm in ("best", "maxpool"):
        better = sum(r < h for r, h in zip(ranks[arm], ranks["head"]))
        worse = sum(r > h for r, h in zip(ranks[arm], ranks["head"]))
        print(f"\n  {arm} vs head: better on {better}, worse on {worse}, "
              f"tied on {n - better - worse}")
        if args.show_flips:
            for (gold, query, _), h, r in zip(pools, ranks["head"], ranks[arm]):
                if h != r:
                    print(f"    {h:>3} -> {r:<3} {query[:60]}")
    con.close()


if __name__ == "__main__":
    main()
