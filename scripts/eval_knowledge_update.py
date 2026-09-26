"""When a fact was corrected, does recall hand the reader the current version?

`mem_write(supersedes=)` marks an old memory with what replaced it, and recall reports
that as `superseded_by` without touching the ranking (`store.supersede`). Whether that
design is enough was never measured. The store now holds the data to measure it with:
every mark a writer made is a real correction, not an authored case.

For each marked memory, follow `superseded_by` to the end of the chain -- the current
version -- and ask recall about it twice: once with the old memory's topic (someone
remembering the thing as it was) and once with the current topic. Both queries echo a
topic, which the lexical lane weights 3x, so absolute recall is optimistic -- and the
relation between stale and current is biased too: a stale-topic query favours the stale
version by construction, a current-topic query the current one. Read the two rows as
bounds, not as one estimate:

- current in window   -- the reader gets the current version at all
- stale first         -- the reader meets the stale version before the current one,
                         either above it or with the current missing; this is the
                         case the mark exists for, since the ranking will not fix it
- stale above current -- both in the window, stale ranked higher
- stale only          -- the stale version is in the window and the current is not;
                         the mark is then the only pointer to what is true now
- hops                -- how many `superseded_by` links the reader must follow from the
                         stale result to reach the current version (the mark resolves
                         one link, so a chain needs a recall per hop)

    python scripts/eval_knowledge_update.py

Runs the production path (gated rerank). Measure on a clean copy with
`AGENT_MEMORY_HOT_PATHS` pointed at a missing directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db, store  # noqa: E402

WINDOW = 5


def chains(con) -> list[tuple[str, list[str]]]:
    """(stale id, [ids from the stale one's successor to the current version])."""
    marked = {
        row["id"]: row["superseded_by"]
        for row in con.execute(
            "SELECT id, superseded_by FROM memories WHERE superseded_by IS NOT NULL")
    }
    out = []
    for stale, ref in marked.items():
        path, seen = [], {stale}
        while ref:
            target = store.resolve_ref(con, ref)
            if target is None or target["id"] in seen:
                path = []  # unresolved or cyclic: nothing to measure against
                break
            path.append(target["id"])
            seen.add(target["id"])
            ref = marked.get(target["id"])
        if path:
            out.append((stale, path))
    return out


def main() -> None:
    con = db.connect()
    pairs = chains(con)
    topic = dict(con.execute("SELECT id, topic FROM memories").fetchall())
    print(f"{len(pairs)} marked memories with a resolvable chain "
          f"(longest {max(len(p) for _, p in pairs)} hops)")

    for label, pick in (("query = stale topic", lambda s, p: topic[s]),
                        ("query = current topic", lambda s, p: topic[p[-1]])):
        n = cur_in = stale_in = first = above = only = both = 0
        hops = []
        for stale, path in pairs:
            current = path[-1]
            ids = [r["id"] for r in store.recall(
                con, pick(stale, path), limit=WINDOW, detail=1)["results"]]
            n += 1
            has_cur, has_stale = current in ids, stale in ids
            cur_in += has_cur
            stale_in += has_stale
            both += has_cur and has_stale
            if has_stale and (not has_cur or ids.index(stale) < ids.index(current)):
                first += 1
                hops.append(len(path))
            above += has_stale and has_cur and ids.index(stale) < ids.index(current)
            only += has_stale and not has_cur
        print(f"\n{label}  (n={n}, window {WINDOW})")
        print(f"  current in window        {cur_in/n:6.1%}")
        print(f"  stale in window          {stale_in/n:6.1%}   both {both/n:6.1%}")
        print(f"  stale first              {first/n:6.1%}   <- the mark is what tells the reader")
        print(f"  stale above current      {above}/{both} of the windows holding both")
        print(f"  stale only               {only/n:6.1%}   <- the mark is the only pointer")
        if hops:
            print(f"  hops to current, stale-first cases: 1 hop {hops.count(1)} · "
                  f"2+ hops {sum(h > 1 for h in hops)} (max {max(hops)})")


if __name__ == "__main__":
    main()
