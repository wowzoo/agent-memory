"""Does a hot section retrieve better when it knows which file it is in?

A hot section is indexed as `heading` + `heading\\n\\nbody` and nothing else, so the
file it belongs to and the headings above it are not searchable at all -- `path` and
`session` are columns, not FTS fields. Two measured problems point here: 14 memory
files share a name across projects, and this project's own hot recall is the worst of
the six (R@1 14.3%) because 40-odd "판정 완료" sections of one file compete.

Anthropic's contextual retrieval reports a 35% cut in top-20 failures from prepending
an LLM-written description to each chunk. This is the deterministic version of the same
idea -- `파일stem > 상위heading > heading` costs no model call and no query latency --
so it is worth knowing whether the free version buys anything before paying for a
generated one.

Two variants, because the sparse and dense lanes want different things (Doc2Query++
measures document expansion helping dense retrieval and *hurting* BM25 through
term-frequency dilution):

    topic         the breadcrumb replaces the heading in the FTS topic column and in
                  the topic vector. `body` FTS is left alone, so no BM25 dilution.
    chunks        as `topic`, and the breadcrumb is also prepended to every body chunk
                  before embedding -- contextual retrieval proper.
    dense         the FTS index is not touched at all; only the topic vector changes.
    dense-chunks  as `dense`, plus the breadcrumb on every body chunk.

The dense pair exists because `topic` was measured first and sells cold hybrid R@5 5.6
points to buy hot R@1 2.6. `memories_fts` holds both tiers, and a file stem is made of
exactly the words people search with (`project`, `memory`, `recall`, `feedback`), so
putting it in a 3x-weighted field lifts every section of that file for any query that
mentions them. The gain in that run was on the semantic lane, so these two variants keep
the gain and leave BM25 alone.

Runs on a **copy** of the store, and rewrites the index in place: `memories` rows keep
their ids, so graph annotations survive. That matters twice -- a forced re-index would
drop 1,200 annotations and the graph lane along with them, which would confound this
measurement, and the in-place shape is what a production migration has to do anyway.

    python scripts/experiment_hot_context.py --variant topic
    python scripts/experiment_hot_context.py --variant chunks --keep
"""

import argparse
import importlib
import os
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIVE_DB = Path(os.environ.get("AGENT_MEMORY_DB", Path.home() / ".agent-memory" / "memory.db"))


def breadcrumbs(file: Path) -> dict[tuple[str, str], str]:
    """(heading, body) -> `stem > ancestors > heading`, keyed exactly as `hot._sections`.

    The key has to be that pair and the body has to include its own heading line,
    because `hot._section_id` hashes that string -- key it on the text after the heading
    and every id misses. Otherwise this mirrors `_sections`, plus the heading level,
    which `_sections` deliberately drops: it flattens h1-h3 because a section is a
    retrieval unit, and the ancestry only matters for this label.
    """
    from agent_memory import hot

    text = hot._FRONTMATTER.sub("", file.read_text(encoding="utf-8", errors="replace"))
    matches = list(hot._HEADING.finditer(text))
    out: dict[tuple[str, str], str] = {}
    if not matches:
        stripped = text.strip()
        if stripped:
            out[(file.stem, stripped)] = file.stem
        return out

    preamble = text[: matches[0].start()].strip()
    if preamble:
        out[(file.stem, preamble)] = file.stem
    ancestors: dict[int, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        level, heading = len(match.group(1)), match.group(2).strip()
        ancestors = {lvl: h for lvl, h in ancestors.items() if lvl < level}
        if body:
            chain = [ancestors[lvl] for lvl in sorted(ancestors)]
            out[(heading, f"{heading}\n\n{body}")] = " > ".join(
                [file.stem, *chain, heading]
            )
        ancestors[level] = heading
    return out


def rewrite(con, variant: str) -> dict:
    """Re-index every hot section with its breadcrumb. Ids and annotations untouched."""
    from agent_memory import embedding, hot, korean

    stats = {"files": 0, "sections": 0, "missing": 0}
    paths = [r["path"] for r in con.execute(
        "SELECT DISTINCT path FROM memories WHERE tier = 'hot' AND path IS NOT NULL"
    )]
    for path in sorted(paths):
        file = Path(path)
        if not file.is_file():
            continue
        stats["files"] += 1
        for (heading, body), crumb in breadcrumbs(file).items():
            memory_id = hot._section_id(file, heading, body)
            row = con.execute(
                "SELECT id FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:  # a section the mirror has not caught up with
                stats["missing"] += 1
                continue
            if not variant.startswith("dense"):
                con.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
                )
                con.execute(
                    "INSERT INTO memories_fts(topic, body, memory_id) VALUES (?, ?, ?)",
                    (korean.index_text(crumb), korean.index_text(body), memory_id),
                )
            con.execute("DELETE FROM memory_chunks WHERE memory_id = ?", (memory_id,))
            chunks = embedding.passages(crumb, body)
            if variant.endswith("chunks"):
                chunks = [chunks[0], *(f"{crumb}\n{c}" for c in chunks[1:])]
            vectors = embedding.encode_passages(chunks)
            con.executemany(
                "INSERT INTO memory_chunks(memory_id, idx, text, embedding) "
                "VALUES (?, ?, ?, ?)",
                [(memory_id, i, t, embedding.to_blob(v))
                 for i, (t, v) in enumerate(zip(chunks, vectors))],
            )
            stats["sections"] += 1
    con.commit()
    return stats


def measure(con, label: str) -> None:
    import json

    from eval_paraphrase import FIXTURE, resolve, resolve_hot
    from agent_memory import store

    print(f"  -- {label}")
    for name, fixture in (("cold 72", FIXTURE),
                          ("hot 38", FIXTURE.parent / "hot_queries.json")):
        cases = []
        for case in json.loads(fixture.read_text())["cases"]:
            gold = (resolve_hot(con, case["path"], case["heading"]) if "path" in case
                    else resolve(con, case["topic_match"]))
            if gold is not None:
                cases.append((gold, case["query"]))
        for mode in ("lexical", "semantic", "hybrid"):
            ranks = []
            for gold, query in cases:
                results = store.recall(con, query, limit=20, detail=1, mode=mode,
                                       rerank=False)["results"]
                ids = [r["id"] for r in results]
                ranks.append(ids.index(gold) + 1 if gold in ids else 21)
            n = len(ranks)
            print(f"     {name:<8}{mode:<9} n={n:<3} "
                  f"R@1={sum(r == 1 for r in ranks)/n:6.1%}  "
                  f"R@5={sum(r <= 5 for r in ranks)/n:6.1%}  "
                  f"MRR={sum(1/r for r in ranks)/n:.3f}  "
                  f"median={int(statistics.median(ranks))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="topic",
                    choices=("topic", "chunks", "dense", "dense-chunks"))
    ap.add_argument("--work", type=Path, default=Path("/tmp/hot_context.db"))
    ap.add_argument("--keep", action="store_true", help="leave the working copy behind")
    args = ap.parse_args()

    # The copy is the whole safety story: this rewrites an index, and doing that to the
    # live store would move every baseline in the repo for every project sharing it.
    for suffix in ("", "-wal", "-shm"):
        src, dst = Path(str(LIVE_DB) + suffix), Path(str(args.work) + suffix)
        if src.exists():
            shutil.copy2(src, dst)
    print(f"working copy: {args.work} (from {LIVE_DB})")

    os.environ["AGENT_MEMORY_DB"] = str(args.work)
    os.environ["AGENT_MEMORY_HOT"] = "0"  # the copy already holds the mirror
    from agent_memory import config, db

    importlib.reload(config)
    importlib.reload(db)
    con = db.connect()
    assert str(config.DB_PATH) == str(args.work), config.DB_PATH

    counts = {r["tier"]: r["n"] for r in con.execute(
        "SELECT tier, COUNT(*) AS n FROM memories GROUP BY tier")}
    entities = con.execute("SELECT COUNT(*) AS n FROM memory_entities").fetchone()["n"]
    print(f"store: hot {counts.get('hot', 0)} : cold {counts.get('cold', 0)}, "
          f"annotations {entities}\n")

    measure(con, "before (heading only)")
    stats = rewrite(con, args.variant)
    after = con.execute("SELECT COUNT(*) AS n FROM memory_entities").fetchone()["n"]
    print(f"\n  rewrote {stats['sections']} sections in {stats['files']} files, "
          f"{stats['missing']} not in the mirror; annotations {entities} -> {after}\n")
    measure(con, f"after (variant={args.variant})")
    con.close()
    if not args.keep:
        for suffix in ("", "-wal", "-shm"):
            Path(str(args.work) + suffix).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
