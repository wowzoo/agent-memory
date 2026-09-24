"""Turn recall queries asked in real sessions into evaluable cases.

The 72-case paraphrase fixture is answered 100% out of cold, and the hot mirror is
now 5.6x the archive (695 sections vs 124 memories), so that set can no longer
judge anything two-tier: tier slots, a default project scope and cross-project
recall all move numbers it cannot see. It needs cases whose answer is a hot
section -- and hand-writing those is exactly what put the optimism bias in the
existing set, because a query written just after reading its target inherits the
target's vocabulary (measured: the first draft of 52 queries scored R@1 84.6%, and
rewriting the 24 most obvious ones took it to 71.2%). A query asked by a session
that did not know the answer yet cannot inherit anything, so they are collected in
the field instead of authored.

Sessions drop one JSON file per logged recall into $AGENT_MEMORY_HOME/recall-log/,
written with the Write tool. Not through this script's argv and not through
mem_write: Korean sent as MCP tool arguments is silently rewritten (94 of 272 new
entity names in one measured run), while a file round-trip preserves it.

A drop looks like this -- `rank` is where the useful answer actually came back, or
null when the response did not contain it at all (the most valuable kind), and
`limit` is the window that recall was asked for:

    {"project": "aidlc-v2-patch", "date": "2026-08-12",
     "query": "물어본 문장 그대로",
     "target": {"tier": "hot", "path": "/Users/.../memory/foo.md",
                "heading": "## 섹션 제목"},
     "rank": null, "limit": 5, "rerank": false, "note": "왜 이게 정답인지 한 줄"}

Cold targets carry `{"tier": "cold", "topic_match": "<topic 앞부분>"}` instead,
which is the key the existing fixture already uses.

    python scripts/collect_queries.py
    python scripts/collect_queries.py --to-cases tests/data/collected_queries.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_memory import config, db  # noqa: E402
from build_hot_queries import tilde  # noqa: E402
from eval_paraphrase import resolve, resolve_hot  # noqa: E402

DROPS = Path(config.HOME) / "recall-log"


def _load(directory: Path) -> list[tuple[Path, dict, str]]:
    """(file, drop, complaint). A malformed drop is reported, never skipped
    silently -- a query lost here is a session's observation that cannot be redone."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            drop = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            out.append((path, {}, f"unreadable: {exc}"))
            continue
        complaint = ""
        if not isinstance(drop, dict):
            complaint = "not a JSON object"
        elif not str(drop.get("query", "")).strip():
            complaint = "no query"
        elif not isinstance(drop.get("target"), dict):
            complaint = "no target"
        out.append((path, drop if isinstance(drop, dict) else {}, complaint))
    return out


def _narrow_window(drop: dict) -> str:
    """`rank: null` is evidence of a failed recall only if the response had room for
    the answer. Asked with limit=2, a target at rank 3 never had a slot -- recording
    that as a miss would teach the fixture that a hit was a miss."""
    if drop.get("rank") is not None:
        return ""
    limit = drop.get("limit")
    if isinstance(limit, int) and limit < config.RECALL_LIMIT:
        return (f"rank is null but limit was {limit} (< {config.RECALL_LIMIT}): "
                "a narrow window, not a miss -- ask again with the full limit")
    return ""


def _moved(con, path: str, heading: str) -> str | None:
    """The same section after its project directory moved.

    A drop records the path as it was when the query was asked, and projects move
    (measured 2026-09-13: four of four collected cases pointed at directories that no
    longer existed). The observation is still valid, so the file name and heading --
    which the move does not change -- are used to find where the section went. Only a
    unique match counts; the same file name under two projects resolves to nothing.
    """
    name = Path(path).name
    # Suffix comparison, not LIKE: these file names contain `_`, which LIKE treats as a
    # single-character wildcard (see the same fix in build_hot_queries.py).
    rows = con.execute(
        "SELECT id FROM memories WHERE tier = 'hot' "
        "AND substr(path, length(path) - ? + 1) = ? AND topic = ?",
        (len(name) + 1, f"/{name}", heading[:120]),
    ).fetchall()
    return rows[0]["id"] if len(rows) == 1 else None


def _resolve(con, target: dict) -> tuple[str | None, str]:
    if target.get("tier") == "hot":
        path, heading = target.get("path"), target.get("heading")
        if not path or not heading:
            return None, "hot target needs both path and heading"
        memory_id = resolve_hot(con, path, heading) or _moved(con, path, heading)
        return memory_id, "" if memory_id else "path+heading matches no single section"
    prefix = target.get("topic_match")
    if not prefix:
        return None, "cold target needs topic_match (or tier: hot)"
    memory_id = resolve(con, prefix)
    return memory_id, "" if memory_id else "topic_match is not a unique prefix"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drops", type=Path, default=DROPS)
    parser.add_argument(
        "--to-cases",
        type=Path,
        default=None,
        help="write the usable drops as an eval fixture (default: only check)",
    )
    args = parser.parse_args()

    if not args.drops.is_dir():
        print(f"no drop directory yet: {args.drops}")
        return
    drops = _load(args.drops)
    if not drops:
        print(f"no drops in {args.drops}")
        return

    con = db.connect()
    usable, rejected = [], []
    for path, drop, complaint in drops:
        if not complaint:
            complaint = _narrow_window(drop)
        if not complaint:
            memory_id, complaint = _resolve(con, drop["target"])
            if memory_id:
                usable.append((path, drop, memory_id))
        if complaint:
            rejected.append((path, drop, complaint))

    print(f"{len(drops)} drops in {args.drops}\n")
    for path, drop, memory_id in usable:
        rank = drop.get("rank")
        print(f"  ok    {drop.get('project', '?'):24} {drop['target'].get('tier', '?'):4} "
              f"{'miss' if rank is None else f'rank {rank}':>7}  {drop['query'][:44]}")
    for path, drop, complaint in rejected:
        print(f"  SKIP  {path.name}: {complaint}")

    tiers = Counter(d["target"].get("tier", "?") for _, d, _ in usable)
    projects = Counter(d.get("project", "?") for _, d, _ in usable)
    queries = Counter(d["query"].strip() for _, d, _ in usable)
    misses = sum(d.get("rank") is None for _, d, _ in usable)
    print(f"\n  usable {len(usable)}  (" + " / ".join(f"{k} {v}" for k, v in sorted(tiers.items()))
          + f")  unusable {len(rejected)}")
    unchecked = sum(
        d.get("rank") is None and not isinstance(d.get("limit"), int)
        for _, d, _ in usable
    )
    print(f"  target not in the response: {misses}"
          + (f"  (⚠ {unchecked} with no `limit` recorded — window unverifiable)"
             if unchecked else ""))
    print("  projects: " + ", ".join(f"{k} {v}" for k, v in sorted(projects.items())))
    repeated = [q for q, n in queries.items() if n > 1]
    if repeated:
        # Two sessions logging the same question is a duplicated case, not two cases:
        # it would count twice in R@1 and quietly weight that memory double.
        print(f"  ⚠ {len(repeated)} query text(s) appear in more than one drop:")
        for query in repeated:
            print(f"      {query[:60]}")

    if args.to_cases is None:
        print("\n  nothing written (pass --to-cases <path> to emit a fixture)")
        return

    cases = []
    for _, drop, memory_id in usable:
        target = drop["target"]
        case = {"query": drop["query"].strip()}
        if target.get("tier") == "hot":
            # The store's coordinates for the section that was resolved, not the drop's.
            # A drop records where the section was when the query was asked, and
            # `_moved` accepts it after the project moved -- but the fixture is read by
            # `resolve_hot`, which matches the path exactly. Writing the drop's path
            # through emits a case that resolves here and nowhere else.
            row = con.execute(
                "SELECT path, topic FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            case["path"] = tilde(row["path"])
            case["heading"] = row["topic"]
        else:
            case["topic_match"] = target["topic_match"]
        cases.append(case)
    args.to_cases.write_text(
        json.dumps(
            {
                "_comment": [
                    "Queries asked in real sessions, collected by scripts/collect_queries.py.",
                    "Kept apart from paraphrase_queries.json on purpose: that set's 72 cases",
                    "are the baseline every recall A/B so far was measured against, and adding",
                    "cases to it would move the denominator under those numbers.",
                    "Hot cases key on path+heading because a hot id is uuid5(path|heading|",
                    "body-hash) and dies the next time that section is edited.",
                ],
                "cases": cases,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    con.close()
    print(f"\n  wrote {len(cases)} cases to {args.to_cases}"
          f"  ({len(rejected)} drop(s) left out — see SKIP above)")


if __name__ == "__main__":
    main()
