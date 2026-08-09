"""Does giving the topic line its own signal improve paraphrase recall?

Baseline hybrid on the 20 hand-authored paraphrase queries is R@1 25.0% /
R@5 75.0% / MRR 0.462. Paraphrase queries usually ask about a memory's *subject*,
and every memory here opens with a keyword-dense topic line -- but that line is
currently buried inside a 400-token chunk 0 and inside one undifferentiated FTS
column, so it carries no more weight than any other sentence.

Two variants, measured against the same fixture and the same RRF settings as
production so only the treatment differs:

  A  semantic: embed the topic as its own vector, max-pooled with the body chunks
  B  lexical:  index topic and body as separate FTS5 columns, weight topic higher
  C  both

Nothing here writes to the store; the experiment builds its own indexes.
"""

import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, embedding, korean  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "paraphrase_queries.json"
DEPTH = 20
TOPIC_WEIGHTS = (3.0, 1.0)  # (topic column, body column) for bm25()


def load():
    con = db.connect()
    memories = con.execute(
        "SELECT id, topic, content FROM memories ORDER BY created_at"
    ).fetchall()
    chunks = con.execute("SELECT memory_id, embedding FROM memory_chunks").fetchall()
    con.close()
    return memories, chunks


def build_lexical(memories):
    """Single-column (as shipped) and two-column (variant B) FTS indexes."""
    lex = sqlite3.connect(":memory:")
    lex.execute("CREATE VIRTUAL TABLE one USING fts5(index_text, tokenize='unicode61')")
    lex.execute(
        "CREATE VIRTUAL TABLE two USING fts5(topic, body, tokenize='unicode61')"
    )
    ids = []
    for row in memories:
        ids.append(row["id"])
        lex.execute("INSERT INTO one(index_text) VALUES (?)",
                    (korean.index_text(row["content"]),))
        lex.execute("INSERT INTO two(topic, body) VALUES (?, ?)",
                    (korean.index_text(row["topic"]), korean.index_text(row["content"])))
    return lex, ids


def lexical_rank(lex, ids, query, two_column: bool) -> list[str]:
    match = korean.match_query(query)
    if not match:
        return []
    if two_column:
        sql = (f"SELECT rowid FROM two WHERE two MATCH ? "
               f"ORDER BY bm25(two, {TOPIC_WEIGHTS[0]}, {TOPIC_WEIGHTS[1]}) LIMIT ?")
        table = "two"
    else:
        sql = "SELECT rowid FROM one WHERE one MATCH ? ORDER BY bm25(one) LIMIT ?"
        table = "one"
    try:
        rows = lex.execute(sql, (match, config.CANDIDATE_POOL)).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"    ! {table}: {exc}")
        return []
    return [ids[r[0] - 1] for r in rows]


def semantic_rank(query_vec, chunk_ids, chunk_matrix, topic_ids, topic_matrix,
                  use_topic: bool) -> list[str]:
    best: dict[str, float] = {}
    for mid, sim in zip(chunk_ids, chunk_matrix @ query_vec):
        if sim > best.get(mid, -1.0):
            best[mid] = float(sim)
    if use_topic:
        for mid, sim in zip(topic_ids, topic_matrix @ query_vec):
            if sim > best.get(mid, -1.0):
                best[mid] = float(sim)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    return [mid for mid, _ in ranked[: config.CANDIDATE_POOL]]


def fuse(*lanes) -> list[str]:
    scores: dict[str, float] = {}
    for lane in lanes:
        for rank, mid in enumerate(lane, start=1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (config.RRF_K + rank)
    return [mid for mid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def report(name, ranks):
    n = len(ranks)
    print(f"  {name:34} R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
          f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
          f"MRR={sum(1/r for r in ranks)/n:.3f}  "
          f"median={int(statistics.median(ranks))}")


def main() -> None:
    memories, chunk_rows = load()
    by_prefix = {row["id"][:8]: row["id"] for row in memories}
    cases = [
        (by_prefix[c["memory_id"]], c["query"])
        for c in json.loads(FIXTURE.read_text())["cases"]
        if c["memory_id"] in by_prefix
    ]
    print(f"{len(cases)} paraphrase cases over {len(memories)} memories\n")

    chunk_ids = [r["memory_id"] for r in chunk_rows]
    chunk_matrix = np.stack([embedding.from_blob(r["embedding"]) for r in chunk_rows])

    topics = [re.sub(r"\s+", " ", row["topic"]).strip() for row in memories]
    topic_ids = [row["id"] for row in memories]
    topic_matrix = embedding.encode_passages(topics)

    lex, ids = build_lexical(memories)

    variants = {
        "baseline (as shipped)": (False, False),
        "A  + topic vector": (False, True),
        "B  + topic column weighted 3x": (True, False),
        "C  both": (True, True),
    }
    results: dict[str, list[int]] = {name: [] for name in variants}
    lanes_only: dict[str, list[int]] = {"lexical two-col": [], "semantic + topic": []}

    for gold, query in cases:
        qv = embedding.encode_query(query)
        lex_one = lexical_rank(lex, ids, query, two_column=False)
        lex_two = lexical_rank(lex, ids, query, two_column=True)
        sem_plain = semantic_rank(qv, chunk_ids, chunk_matrix, topic_ids, topic_matrix, False)
        sem_topic = semantic_rank(qv, chunk_ids, chunk_matrix, topic_ids, topic_matrix, True)

        def rank_of(order):
            return order.index(gold) + 1 if gold in order[:DEPTH] else DEPTH + 1

        for name, (two_col, use_topic) in variants.items():
            order = fuse(lex_two if two_col else lex_one,
                         sem_topic if use_topic else sem_plain)
            results[name].append(rank_of(order))
        lanes_only["lexical two-col"].append(rank_of(lex_two))
        lanes_only["semantic + topic"].append(rank_of(sem_topic))

    print("fused:")
    for name in variants:
        report(name, results[name])
    print("\nsingle lanes (for attribution):")
    for name, ranks in lanes_only.items():
        report(name, ranks)


if __name__ == "__main__":
    main()
