"""Rescue a lane's first *uncorroborated* hit, not only its first hit. Sweeps the depth.

`_rescue_lane_leaders` promotes `ids[0]` and nothing else, so it cannot reach a memory the
lane is certain about that sits deeper -- which is exactly where the noun-join experiment
stopped: with the join, the lexical lane held the gold 15th of 129 matches and rescue could
not see it (⇒ `experiment_compound_join.py`).

The generalization keeps every narrow property of the original and changes only *where in
the lane it looks*: still the two base lanes, still one promotion each, still only rows no
other lane retrieved, still appended at the window's end. `depth=1` is the shipped rule
exactly -- if `ids[0]` is corroborated the scan finds nothing and stops -- so the sweep
carries its own equivalence check in its first row.

    AGENT_MEMORY_HOT=0 AGENT_MEMORY_DB=/tmp/c3.db \
        python scripts/experiment_rescue_depth.py
    ... --depths 1,3,5,10 --join      # with the noun join, its stated resumption condition

⚠️ Deeper is not obviously better. A lane's 10th choice that no other lane found is as
likely to be that lane's noise as its insight, and every promotion **displaces a row the
fusion did agree on**. That is the trade the sweep has to price.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, korean, store  # noqa: E402
from diagnose_candidates import cases_of, position  # noqa: E402
from diagnose_vocabulary import POPULATIONS, WINDOW  # noqa: E402
from experiment_compound_join import tokens_joined  # noqa: E402


def rescue_depth(ranked, lanes, limit, depth):
    """`_rescue_lane_leaders` with the leader test replaced by a bounded scan."""
    if limit <= 1 or len(ranked) <= limit:
        return ranked, 0
    rescued = 0
    for name in ("lexical", "semantic"):
        ids = lanes.get(name) or []
        pick = next(
            (c for c in ids[:depth]
             if sum(c in found for found in lanes.values()) == 1),
            None,
        )
        if pick is None:
            continue
        at = next((i for i, (mid, _) in enumerate(ranked) if mid == pick), None)
        if at is None or at < limit:
            continue
        slot = limit - 1 - rescued
        if slot < 1:
            break
        row = ranked[at]
        ranked = (ranked[:slot] + ranked[slot + 1 : limit] + [row, ranked[slot]]
                  + ranked[limit:at] + ranked[at + 1 :])
        rescued += 1
    return ranked, rescued


def ranks(con) -> dict[str, int]:
    out = {}
    for name, fixture in POPULATIONS:
        for gold, query in cases_of(con, fixture):
            results = store.recall(con, query, limit=WINDOW, detail=1, mode="hybrid",
                                   rerank=None)["results"]
            out[f"{name}|{query}"] = position([r["id"] for r in results], gold) or WINDOW + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,3,5,10")
    ap.add_argument("--join", action="store_true",
                    help="also apply the noun join, rebuilding the index on the copy")
    args = ap.parse_args()

    if config.DB_PATH == Path("~/.agent-memory/memory.db").expanduser():
        sys.exit("REFUSING: point AGENT_MEMORY_DB at a copy (scripts/make_clean_copy.py)")

    con = db.connect()
    print(f"store: {config.DB_PATH}, join={args.join}, "
          f"rescue={config.RESCUE_LANE_LEADERS}, pool={config.CANDIDATE_POOL}")

    if args.join:
        korean.tokens = tokens_joined
        for row in con.execute("SELECT id, topic, content FROM memories").fetchall():
            con.execute("DELETE FROM memories_fts WHERE memory_id = ?", (row["id"],))
            con.execute(
                "INSERT INTO memories_fts(topic, body, memory_id) VALUES (?, ?, ?)",
                (korean.index_text(row["topic"]), korean.index_text(row["content"]),
                 row["id"]),
            )
        con.commit()

    shipped = store._rescue_lane_leaders
    rows = {}
    try:
        for depth in [int(d) for d in args.depths.split(",")]:
            store._rescue_lane_leaders = (
                lambda r, l, lim, _d=depth: rescue_depth(r, l, lim, _d)
            )
            rows[depth] = ranks(con)
            line = f"  depth {depth:>2}"
            for name, _ in POPULATIONS:
                rs = [v for k, v in rows[depth].items() if k.startswith(f"{name}|")]
                line += (f"   {name} R@1={sum(r == 1 for r in rs)/len(rs):5.1%}"
                         f" R@5={sum(r <= WINDOW for r in rs)/len(rs):5.1%}")
            print(line)
    finally:
        store._rescue_lane_leaders = shipped

    base = min(rows)
    for depth in sorted(rows):
        if depth == base:
            continue
        changed = [(k, rows[base][k], rows[depth][k])
                   for k in rows[base] if rows[base][k] != rows[depth][k]]
        print(f"\n  depth {base} -> {depth}: {len(changed)}건 변화")
        for k, x, y in sorted(changed, key=lambda t: t[2] - t[1]):
            pop, query = k.split("|", 1)
            print(f"    {'개선' if y < x else '악화'} {x}->{y}  [{pop}] {query[:48]}")
    con.close()


if __name__ == "__main__":
    main()
