"""How often does the query share no usable word with its answer? Sizes the alias class.

The candidate-generation diagnosis (`diagnose_candidates.py`) ended by splitting the
unreachable residue in two: cases the fusion buried, and cases where the lexical lane
never had a chance because the query and the memory use different words for the same
thing -- "크로스 인코더" asked of a memory that says "리랭커"/"ko-reranker". It named two
such cases and stopped, which is too small a sample to design against. This counts them.

Deliberately no lanes and no ranking: it compares the query's tokens against the *gold's
own* text, so the number does not move when this repository records its own evaluations
into the hot tier (see the measurement-hygiene sections of `docs/measurements.md`). That is
the whole reason this is a separate script -- it is the one question about retrieval that
can be asked on the live store.

`df` is what decides reachability, not the raw count of shared words. BM25 matches on OR,
so a gold that shares zero query terms is not in the match set at all; one that shares
only `pr`(20%) and `올리`(8%) is in a match set of thousands with nothing to lift it. The
bucket is therefore keyed on the *rarest* shared term.

    python scripts/diagnose_vocabulary.py
    python scripts/diagnose_vocabulary.py --show-terms   # and the unmatched-term pairs
    AGENT_MEMORY_DB=/tmp/agent-memory-clean.db \
        python scripts/diagnose_vocabulary.py --rank      # does the bucket actually fail?

`--show-terms` prints the design input: query terms the gold does not contain, beside the
gold's own heading. Alias candidates are read off that pair.

⚠️ `--rank` is the only part that ranks, so it is the only part that needs
`scripts/make_clean_copy.py` first -- on the live store this repository's own records
answer their own queries. It is also the part that decides whether any of this is worth a
device: a query can share nothing usable with its answer and still be recalled at rank 1,
because the embedding lane never needed the words.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, korean, store  # noqa: E402
from diagnose_candidates import HOT_FIXTURE, cases_of, position  # noqa: E402
from eval_paraphrase import FIXTURE  # noqa: E402

# The caller's real window. Ranking outside it is not a delivered answer, and the whole
# point of the buckets is which ones the caller never sees.
WINDOW = 5

# A term this common cannot lift a gold out of a match set: `pr` at 20% is the cold case
# the diagnosis flagged as unreachable even though it technically shares two terms.
COMMON = 0.05

# The third population is the one that matters for any device that rewards lexical
# overlap: `collected_queries.json` is what people actually typed, kept out of the two
# fixtures on purpose so it cannot move their denominators. Every query in the other two
# was written by someone who had read the memory it is supposed to find, so a lexical
# device measured only there is measured on its own best case. Reported separately rather
# than merged, for the same reason it is stored separately.
COLLECTED = FIXTURE.parent / "collected_queries.json"
POPULATIONS = (("cold", FIXTURE), ("hot", HOT_FIXTURE), ("실사용", COLLECTED))


def df_lookup(con, cache: dict[str, int], term: str) -> int:
    if term not in cache:
        try:
            cache[term] = con.execute(
                "SELECT COUNT(*) AS n FROM memories_fts WHERE memories_fts MATCH ?",
                (f'"{term}"',),
            ).fetchone()["n"]
        except Exception:
            cache[term] = -1
    return cache[term]


def survey(con, cases: list[tuple[str, str]], total: int, cache: dict,
           rank: bool) -> list[dict]:
    rows = []
    for gold, query in cases:
        row = con.execute(
            "SELECT topic, content FROM memories WHERE id = ?", (gold,)
        ).fetchone()
        gold_terms = set(korean.tokens(f"{row['topic']}\n{row['content']}"))
        q_terms = list(dict.fromkeys(korean.tokens(query)))
        shared = [t for t in q_terms if t in gold_terms]
        dfs = {t: df_lookup(con, cache, t) / total for t in shared}
        rarest = min(dfs.values()) if dfs else None
        rows.append({
            "query": query,
            "topic": row["topic"],
            "q_terms": q_terms,
            "shared": shared,
            "dfs": dfs,
            "rarest": rarest,
            "missing": [t for t in q_terms if t not in gold_terms],
            "topic_terms": list(dict.fromkeys(korean.tokens(row["topic"]))),
            "bucket": ("공유 없음" if rarest is None
                       else "흔한 것만" if rarest > COMMON
                       else "건강"),
            "rank": _rank(con, query, gold) if rank else None,
        })
    return rows


def _rank(con, query: str, gold: str) -> int:
    """Production path, real window: `rerank=None` follows the config as the server does."""
    results = store.recall(con, query, limit=WINDOW, detail=1, mode="hybrid",
                           rerank=None)["results"]
    return position([r["id"] for r in results], gold) or WINDOW + 1


def report(name: str, rows: list[dict], show_terms: bool) -> None:
    n = len(rows)
    ranked = rows[0]["rank"] is not None if rows else False
    print(f"\n=== {name}: {n} cases")
    for bucket in ("공유 없음", "흔한 것만", "건강"):
        hit = [r for r in rows if r["bucket"] == bucket]
        line = f"    {bucket:<8} {len(hit):>3} / {n} = {len(hit)/n:5.1%}"
        if ranked and hit:
            ranks = [r["rank"] for r in hit]
            line += (f"   R@1={sum(r == 1 for r in ranks)/len(hit):5.1%}"
                     f"  R@5={sum(r <= WINDOW for r in ranks)/len(hit):5.1%}")
        print(line)

    dead = [r for r in rows if r["bucket"] != "건강"]
    if not dead:
        return
    print(f"\n  렉시컬이 닿을 수 없거나 묻히는 {len(dead)}건:")
    for r in sorted(dead, key=lambda r: (r["rarest"] is not None, r["rarest"] or 0)):
        share = " ".join(f"{t}({r['dfs'][t]:.0%})" for t in r["shared"]) or "-"
        at = f" -> rank {r['rank']}" if r["rank"] is not None else ""
        print(f"    [{r['bucket']}] {r['query'][:46]}{at}")
        print(f"        공유: {share}")
        if show_terms:
            print(f"        정답 제목: {r['topic'][:66]}")
            print(f"        안 맞은 질의어: {' '.join(r['missing'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-terms", action="store_true")
    ap.add_argument("--rank", action="store_true",
                    help="also recall each case; needs a clean copy (see module docstring)")
    args = ap.parse_args()

    con = db.connect()
    total = con.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    print(f"store: {config.DB_PATH}, {total} memories, 흔한 용어 기준 df > {COMMON:.0%}")
    cache: dict[str, int] = {}
    for name, fixture in POPULATIONS:
        rows = survey(con, cases_of(con, fixture), total, cache, args.rank)
        report(name, rows, args.show_terms)
    con.close()


if __name__ == "__main__":
    main()
