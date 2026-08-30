"""Joining space-separated nouns — 채택 안 함. Fixes candidate generation, not delivery.

The vocabulary census (`diagnose_vocabulary.py`) found one hot case sharing *no* usable
term with its answer, and reading it showed the cause is not a missing synonym: the memory
writes "크로스인코더" and the query writes "크로스 인코더". Kiwi keeps the closed compound whole
(`split_complex` does not touch it -- checked) and its own guesses are wrong where it does
split ("후보생성" -> 후보생 + 성), so splitting needs a dictionary and joining needs nothing:
emit `크로스인코더` alongside `크로스` and `인코더` whenever two nouns are separated by exactly one
space. `tokens()` already doubles the index this way for the mirror-image problem, so this
is the same trade in the same place.

Measured 2026-08-30 on the contamination-free copy, `limit=5`, `--rerank auto`:

    |        | cold 72 R@1 / R@5 | hot 38 R@1 / R@5 | 실사용 4 R@1 / R@5 |
    | 기준선 | 63.9 / 79.2       | 36.8 / 84.2      | 0.0 / 50.0         |
    | 결합   | 66.7 / 80.6       | 34.2 / 86.8      | 0.0 / 25.0         |

Not adopted. 14 of 114 cases move, 7 up and 7 down, for +2 R@5 on the fixtures at +22.2%
index tokens -- and the case that motivated it is still not delivered. It goes from **rank
>20 to 6**: the join does put the gold in the lexical lane (15th of 129 matches, where
before it matched nothing), and then the fusion buries it, because the semantic lane has it
at 206/1521 and one lane's 15th place loses to two lanes agreeing. That is the pathology
the candidate-generation diagnosis already named, arriving from a new direction.

🔴 **The real-usage column is what settles it.** What this device actually is, is a
noun-bigram index, not compound normalization -- it also produced `cliv` and `칸전수`. On the
fixtures that is harmless, because a junk join only matches a memory containing the same
junk, which is a shared phrase. But every fixture query was written by someone who had
already read the memory it should find, so a device rewarding lexical overlap is measured
there on its own best case. On the four queries people actually typed it **halves R@5**:
"내 이슈 로스터 open PR CONFLICTING #715 #555" goes 3 -> 6. Real questions are keyword
lists, and joining adjacent keywords fabricates terms that were never phrases. n=4 decides
nothing on its own, but it is the only population that is not self-referential and its sign
is the opposite one.

⇒ 재개 조건: extend the lane-leader rescue past `ids[0]`. Rescue promotes only a lane's
*first* choice, so it cannot reach a gold sitting 15th in the lane that is certain about
it. Fix that and this device becomes what puts the gold in reach at all, at which point
the 22% is buying something.

    python scripts/experiment_compound_join.py                  # bucket migration only
    python scripts/experiment_compound_join.py --cost
    AGENT_MEMORY_DB=/tmp/clean.db python scripts/experiment_compound_join.py --eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, korean, store  # noqa: E402
from diagnose_candidates import cases_of, position  # noqa: E402
from diagnose_vocabulary import COMMON, POPULATIONS, WINDOW, df_lookup  # noqa: E402

# Only nouns join. Verb and adverb stems next to each other are a sentence, not a compound.
_NOUNS = frozenset({"NNG", "NNP", "SL", "SH"})
_base_tokens = korean.tokens


def joins(text: str) -> list[str]:
    """Concatenations of two nouns separated by exactly one space, in order.

    The one-space test is what keeps this narrow. Joining any two adjacent nouns would
    also fuse across punctuation and line breaks, where the writer never meant a compound.
    """
    out: dict[str, None] = {}
    toks = [t for t in korean._analyzer().tokenize(text) if t.tag in _NOUNS]
    for a, b in zip(toks, toks[1:]):
        if text[a.start + a.len : b.start] != " ":
            continue
        form = (a.form + b.form).lower()
        if korean._keep(form):
            out[form] = None
    return list(out)


def tokens_joined(text: str) -> list[str]:
    return list(dict.fromkeys(_base_tokens(text) + joins(text)))


def bucket_of(rarest: float | None) -> str:
    return ("공유 없음" if rarest is None
            else "흔한 것만" if rarest > COMMON else "건강")


def migration(con, total: int, cache: dict) -> None:
    """Which cases gain a rare shared term. A proxy for --eval, and much cheaper."""
    for name, fixture in POPULATIONS:
        moved, stayed = [], 0
        for gold, query in cases_of(con, fixture):
            row = con.execute(
                "SELECT topic, content FROM memories WHERE id = ?", (gold,)
            ).fetchone()
            text = f"{row['topic']}\n{row['content']}"
            before = [t for t in _base_tokens(query) if t in set(_base_tokens(text))]
            after = [t for t in tokens_joined(query) if t in set(tokens_joined(text))]

            def rarest(shared):
                dfs = [df_lookup(con, cache, t) / total for t in shared]
                return min(dfs) if dfs else None

            b0, b1 = bucket_of(rarest(before)), bucket_of(rarest(after))
            if b0 == b1:
                stayed += 1
            else:
                moved.append((b0, b1, query, [t for t in after if t not in before]))
        print(f"\n=== {name}: {len(moved)} moved, {stayed} unchanged")
        for b0, b1, query, gained in moved:
            print(f"    {b0} -> {b1}  {query[:44]}")
            print(f"        새 공유어: {' '.join(gained)}")


def ranks(con) -> dict[str, int]:
    out = {}
    for name, fixture in POPULATIONS:
        for gold, query in cases_of(con, fixture):
            results = store.recall(con, query, limit=WINDOW, detail=1, mode="hybrid",
                                   rerank=None)["results"]
            out[f"{name}|{query}"] = position([r["id"] for r in results], gold) or WINDOW + 1
    return out


def score(tag: str, got: dict[str, int]) -> None:
    for name, _ in POPULATIONS:
        rs = [v for k, v in got.items() if k.startswith(f"{name}|")]
        print(f"  {tag:<6} {name:<5} n={len(rs):<3} "
              f"R@1={sum(r == 1 for r in rs)/len(rs):5.1%}  "
              f"R@5={sum(r <= WINDOW for r in rs)/len(rs):5.1%}")


def evaluate(con) -> None:
    """One process, one store, two index states -- so only the tokenizer differs."""
    before = ranks(con)
    score("기준선", before)

    korean.tokens = tokens_joined  # index and query must agree; both call this
    rows = con.execute("SELECT id, topic, content FROM memories").fetchall()
    for row in rows:
        con.execute("DELETE FROM memories_fts WHERE memory_id = ?", (row["id"],))
        con.execute(
            "INSERT INTO memories_fts(topic, body, memory_id) VALUES (?, ?, ?)",
            (korean.index_text(row["topic"]), korean.index_text(row["content"]), row["id"]),
        )
    con.commit()
    after = ranks(con)
    score("결합", after)

    changed = [(k, before[k], after[k]) for k in before if before[k] != after[k]]
    print(f"\n  변화 {len(changed)} / {len(before)}건")
    for k, x, y in sorted(changed, key=lambda t: t[2] - t[1]):
        tier, query = k.split("|", 1)
        print(f"    {'개선' if y < x else '악화'} {x}->{y}  [{tier}] {query[:50]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--eval", action="store_true",
                    help="rebuild the index on a copy and measure; needs AGENT_MEMORY_DB")
    args = ap.parse_args()

    if args.eval and config.DB_PATH == Path("~/.agent-memory/memory.db").expanduser():
        sys.exit("REFUSING: --eval rewrites the index; point AGENT_MEMORY_DB at a copy "
                 "(scripts/make_clean_copy.py)")

    con = db.connect()
    total = con.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    print(f"store: {config.DB_PATH}, {total} memories")

    if args.eval:
        evaluate(con)
    else:
        migration(con, total, {})

    if args.cost:
        base = extra = 0
        for row in con.execute("SELECT topic, content FROM memories"):
            text = f"{row['topic']}\n{row['content']}"
            base += len(_base_tokens(text))
            extra += len(joins(text))
        print(f"\n=== 색인 비용: 토큰 {base} -> {base + extra} (+{extra/base:.1%})")

    con.close()


if __name__ == "__main__":
    main()
