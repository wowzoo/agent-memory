"""Is a cross-encoder reranker worth its latency? Ceiling first, then models.

A reranker reorders the fused candidate list and cannot introduce a memory the base
lanes never retrieved, so its ceiling is fixed by how often the answer is already in
the top N. Measuring that before exporting anything is cheaper than finding out
afterwards that there were three points to play for.

    python scripts/experiment_rerank.py --tier cold                 # ceiling only
    python scripts/experiment_rerank.py --tier cold --model ~/.agent-memory/models/ko-reranker

Ceiling table: `in pool` is what a perfect reranker would score at R@1 and R@5, since
it would place the answer first. `recoverable` counts today's misses (rank > 5) that
sit inside the pool and are therefore a reranker's actual targets.

With --model, the fused top `--depth` is re-scored by the cross-encoder and the
metrics are recomputed. The passage is the memory's topic plus its body, truncated by
the tokenizer to the model's window -- one pass per candidate, so a long memory is
judged on its opening. Latency is reported per query because that is what the cost
argument turns on: the base lanes answer in about 10 ms.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, store  # noqa: E402
from eval_paraphrase import FIXTURE, resolve  # noqa: E402

DEPTHS = (5, 10, 20, 50)
POOL = max(DEPTHS)
BATCH = 16


def load(model_dir: Path, max_tokens: int):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    # Never inherit the file's truncation/padding settings. A re-export once baked
    # truncation.max_length into tokenizer.json and silently cut indexing inputs
    # (599 chunks became 337); the fix was to state both explicitly at load.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=max_tokens)
    tokenizer.enable_padding()
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    return tokenizer, session, {i.name for i in session.get_inputs()}


def rerank_scores(model, query: str, passages: list[str]) -> list[float]:
    import numpy as np

    tokenizer, session, names = model
    out: list[float] = []
    for start in range(0, len(passages), BATCH):
        chunk = passages[start:start + BATCH]
        encodings = tokenizer.encode_batch([(query, p) for p in chunk])
        feed = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64)}
        if "attention_mask" in names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )
        out.extend(session.run(None, feed)[0].reshape(-1).tolist())
    return out


def report(label: str, ranks: list[int]) -> None:
    n = len(ranks)
    print(f"  {label:<12}R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
          f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
          f"MRR={sum(1/r for r in ranks)/n:.3f}  "
          f"median={int(statistics.median(ranks))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("hot", "cold"), default=None)
    parser.add_argument("--mode", default="hybrid",
                        choices=("hybrid", "lexical", "semantic"))
    parser.add_argument("--model", type=Path, default=None,
                        help="exported reranker directory; omit for the ceiling only")
    parser.add_argument("--depth", type=int, default=20,
                        help="how many fused candidates the reranker re-scores")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="query+passage window; shorter is faster and may be no worse")
    args = parser.parse_args()

    cases = json.loads(FIXTURE.read_text())["cases"]
    con = db.connect()
    resolved = [
        (full, case["query"])
        for case in cases
        if (full := resolve(con, case["topic_match"])) is not None
    ]
    n = len(resolved)
    print(f"{n} paraphrase cases, mode={args.mode}, tier={args.tier or 'both'}\n")

    # One retrieval pass, reused for both the ceiling and the reranking, so the two
    # cannot disagree about what the base lanes returned.
    pools = []
    for memory_id, query in resolved:
        results = store.recall(
            con, query, limit=POOL, detail=3, mode=args.mode, tier=args.tier
        )["results"]
        pools.append((memory_id, query, results))

    base_ranks = []
    for memory_id, _, results in pools:
        ids = [r["id"] for r in results]
        base_ranks.append(ids.index(memory_id) + 1 if memory_id in ids else POOL + 1)
    report("base", base_ranks)

    print()
    misses = [r for r in base_ranks if r > 5]
    for depth in DEPTHS:
        in_pool = sum(r <= depth for r in base_ranks)
        recoverable = sum(5 < r <= depth for r in misses)
        print(f"  depth {depth:>3}   in pool {in_pool:>3}/{n} = {in_pool/n:6.1%}"
              f"  → ceiling R@1/R@5 {in_pool/n:6.1%}"
              f"   recoverable misses {recoverable}/{len(misses)}")

    if args.model is None:
        con.close()
        return

    model = load(args.model.expanduser(), args.max_tokens)
    print(f"\nreranking top {args.depth} with {args.model.name} at {args.max_tokens} tokens")
    ranks, elapsed, rerank_orders = [], [], []
    for memory_id, query, results in pools:
        head = results[: args.depth]
        passages = [f"{r['topic']}\n{r['content']}" for r in head]
        start = time.perf_counter()
        scores = rerank_scores(model, query, passages)
        elapsed.append(time.perf_counter() - start)
        # Tie-break on id, as store.recall does, so equal scores do not order by
        # whichever lane happened to see the memory first.
        order = sorted(zip(scores, [r["id"] for r in head]), key=lambda s: (-s[0], s[1]))
        ids = [mid for _, mid in order] + [r["id"] for r in results[args.depth:]]
        rerank_orders.append([mid for _, mid in order])
        ranks.append(ids.index(memory_id) + 1 if memory_id in ids else POOL + 1)
    report("reranked", ranks)
    better = sum(r < b for r, b in zip(ranks, base_ranks))
    worse = sum(r > b for r, b in zip(ranks, base_ranks))
    print(f"  vs base: better on {better}, worse on {worse}, tied on {n - better - worse}")
    print(f"  latency: {statistics.mean(elapsed)*1000:.0f} ms/query mean, "
          f"{statistics.median(elapsed)*1000:.0f} ms median, "
          f"{max(elapsed)*1000:.0f} ms max  (base lanes ≈ 10 ms)")

    # Replacing the fused order throws the base signal away. Blend instead: the same
    # reciprocal-rank fusion the lanes already use, with the reranker as one more
    # lane. Scores are cached above, so the sweep costs nothing.
    print("\n  RRF blend (base order + reranker order):")
    for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        blended = []
        for (memory_id, _, results), rr in zip(pools, rerank_orders):
            head_ids = [r["id"] for r in results[: args.depth]]
            base_rank = {mid: i + 1 for i, mid in enumerate(head_ids)}
            rr_rank = {mid: i + 1 for i, mid in enumerate(rr)}
            fused = {
                mid: 1 / (config.RRF_K + base_rank[mid])
                + weight / (config.RRF_K + rr_rank[mid])
                for mid in head_ids
            }
            order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
            ids = [mid for mid, _ in order] + [r["id"] for r in results[args.depth:]]
            blended.append(ids.index(memory_id) + 1 if memory_id in ids else POOL + 1)
        report(f"w={weight}", blended)
    con.close()


if __name__ == "__main__":
    main()
