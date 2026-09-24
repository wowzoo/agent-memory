"""Conviction weight β, re-measured on the path that actually ships.

β was deferred twice and the reason changed both times: first "it loses one R@5 case",
then -- when that loss did not reproduce -- "it gives back exactly the R@5 the graph lane
bought". Both verdicts came from `experiment_r5_priority.py`, which reimplements fusion in
its own `order()` and therefore has **no reranker, no lane rescue, `limit=20`, and a tier
reservation that is inert at that window**. Every one of those is now part of the shipped
path, so that harness can no longer re-measure its own conclusion.

This measures β where it would actually live: `store._fuse` patched, everything downstream
untouched, `store.recall(limit=5, rerank=None)` so the gate, the reranker, the rescue and
`MIN_TIER_SLOTS` all apply. `β=0` is plain RRF by definition, so the first row is the
shipped baseline and its agreement with the recorded numbers is the equivalence check.

The question is narrow and worth asking: β rewards one lane being sure, and so does the
gated reranker (it fires precisely when the lanes disagree about first place, ⇒
`docs/measurements.md` `### 리랭커를 켤지는 베이스 순위가 먼저 압니다`), and so does the lane
rescue. Three devices
aimed at one pathology. If β still pays after the other two, it is buying something they
do not; if it does not, that is the fourth time a broad device handed its job to a narrow
one and the axis closes.

    AGENT_MEMORY_HOT=0 AGENT_MEMORY_DB=/tmp/c4.db python scripts/experiment_conviction.py
    ... --betas 0,0.25,0.5,1.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config, db, store  # noqa: E402
from diagnose_candidates import cases_of, position  # noqa: E402
from diagnose_vocabulary import POPULATIONS, WINDOW  # noqa: E402

_base_fuse = store._fuse


def fuse_with_beta(lanes, k, weights=None, beta=0.0):
    """`_fuse` plus the single best lane contribution, which is what β means.

    Kept as an addition to the summed score rather than a replacement, so β=0 is
    bit-identical to the shipped function instead of merely close to it.
    """
    scores = _base_fuse(lanes, k, weights)
    if not beta:
        return scores
    best: dict[str, float] = {}
    for name, ids in sorted(lanes.items()):
        w = (weights or {}).get(name, 1.0)
        for rank, mid in enumerate(ids, start=1):
            c = w / (k + rank)
            if c > best.get(mid, 0.0):
                best[mid] = c
    return {m: s + beta * best[m] for m, s in scores.items()}


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
    ap.add_argument("--betas", default="0,0.25,0.5,1.0")
    args = ap.parse_args()

    if config.DB_PATH == Path("~/.agent-memory/memory.db").expanduser():
        sys.exit("REFUSING: point AGENT_MEMORY_DB at a copy (scripts/make_clean_copy.py)")

    con = db.connect()
    print(f"store: {config.DB_PATH}, graph w={config.GRAPH_LANE_WEIGHT}, "
          f"RRF_K={config.RRF_K}, rerank={config.RERANK_ENABLED}/gate={config.RERANK_GATE}, "
          f"rescue={config.RESCUE_LANE_LEADERS}, slots={config.MIN_TIER_SLOTS}")

    rows = {}
    try:
        for beta in [float(b) for b in args.betas.split(",")]:
            store._fuse = lambda l, k, w=None, _b=beta: fuse_with_beta(l, k, w, _b)
            rows[beta] = ranks(con)
            line = f"  β={beta:<5}"
            for name, _ in POPULATIONS:
                rs = [v for k, v in rows[beta].items() if k.startswith(f"{name}|")]
                line += (f"   {name} R@1={sum(r == 1 for r in rs)/len(rs):5.1%}"
                         f" R@5={sum(r <= WINDOW for r in rs)/len(rs):5.1%}"
                         f" MRR={sum(1/r for r in rs)/len(rs):.3f}")
            print(line)
    finally:
        store._fuse = _base_fuse

    base = min(rows)
    for beta in sorted(rows):
        if beta == base:
            continue
        changed = [(k, rows[base][k], rows[beta][k])
                   for k in rows[base] if rows[base][k] != rows[beta][k]]
        print(f"\n  β {base} -> {beta}: {len(changed)}건 변화")
        for k, x, y in sorted(changed, key=lambda t: t[2] - t[1]):
            pop, query = k.split("|", 1)
            print(f"    {'개선' if y < x else '악화'} {x}->{y}  [{pop}] {query[:48]}")
    con.close()


if __name__ == "__main__":
    main()
