"""Write and recall — the two operations that carried 86% of real MARM usage."""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config, db, embedding, graph, hot, korean, rerank

_DETAIL_CHARS = {1: 200, 2: 500}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_project() -> str | None:
    explicit = os.environ.get("AGENT_MEMORY_PROJECT", "").strip()
    if explicit:
        return explicit
    cwd = os.getcwd()
    return os.path.basename(cwd) or None


def active_session(con: sqlite3.Connection) -> str:
    return db.get_meta(con, "active_session") or "default"


# --------------------------------------------------------------------------- write

# An `entities` argument that ended up inside `content` instead of beside it.
_ENTITY_ARRAY = re.compile(r'\[\s*\{\s*"name"\s*:.*?\}\s*\]', re.DOTALL)


def _entity_json_in(content: str) -> str | None:
    """The entity array, if the caller pasted it into the prose.

    Measured: five memories in one project carry their own entity list inside their
    text -- the same array they also passed as `entities` -- because the two
    arguments were composed as one string. Nothing is lost, which is why it went
    three weeks unnoticed: the graph took the entities and the prose is intact. The
    cost is that the last chunk of such a memory embeds JSON rather than Korean, and
    its entity labels are counted twice in the lexical lane.

    Reported, not stripped. Cutting text out of what the caller asked to store is a
    worse failure than saying so, and a memory may legitimately quote this shape.
    """
    for match in _ENTITY_ARRAY.finditer(content):
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(parsed, list) and any(
            isinstance(item, dict) and "name" in item for item in parsed
        ):
            return match.group(0)
    return None


def write(
    con: sqlite3.Connection,
    content: str,
    session: str | None = None,
    topic: str | None = None,
    project: str | None = None,
    entities: list | None = None,
    relations: list | None = None,
    supersedes: str | None = None,
) -> dict:
    content = content.strip()
    if not content:
        return {"status": "error", "message": "content is empty"}

    if session:
        db.set_meta(con, "active_session", session)
    else:
        session = active_session(con)
    topic = (topic or content.splitlines()[0])[:120].strip()
    project = project if project is not None else detect_project()

    memory_id = uuid.uuid4().hex
    con.execute(
        "INSERT INTO memories(id, session, project, topic, content, created_at, source) "
        "VALUES (?, ?, ?, ?, ?, ?, 'write')",
        (memory_id, session, project, topic, content, _now()),
    )
    con.execute(
        "INSERT INTO memories_fts(topic, body, memory_id) VALUES (?, ?, ?)",
        (korean.index_text(topic), korean.index_text(content), memory_id),
    )

    chunks = embedding.passages(topic, content)
    vectors = embedding.encode_passages(chunks)
    con.executemany(
        "INSERT INTO memory_chunks(memory_id, idx, text, embedding) VALUES (?, ?, ?, ?)",
        [
            (memory_id, i, text, embedding.to_blob(vec))
            for i, (text, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    con.commit()
    result = {
        "status": "ok",
        "memory_id": memory_id,
        "session": session,
        "topic": topic,
        "project": project,
        "chunks": len(chunks),
    }
    if entities or relations:
        attached = graph.attach(con, memory_id, entities, relations)
        result["graph"] = {
            "entities": attached.get("entities", 0),
            "relations": attached.get("relations", 0),
        }
        # Only when they disagree, so the extra keys mean something: some input did
        # not land one-for-one. Two names that differ by case or spacing merge into
        # one entity and a blank name is skipped -- which is what "I supplied 8 and
        # got 7 back, cause unknown" turned out to be.
        for kind in ("entities", "relations"):
            given = attached.get(f"{kind}_given", 0)
            if given != result["graph"][kind]:
                result["graph"][f"{kind}_given"] = given
    if supersedes:
        # After the insert, so the correction is what the old row points at. Reported
        # rather than raised: the memory is already stored and losing it because a
        # reference did not resolve would be the worse failure.
        marked = supersede(con, supersedes, memory_id)
        result["supersedes"] = (
            marked["superseded"] if marked["status"] == "ok" else marked["message"]
        )
    stray = _entity_json_in(content)
    if stray:
        result["warning"] = (
            f"content contains an entity-shaped JSON array ({len(stray)} chars). If "
            "that is the `entities` argument pasted into the text, the prose kept a "
            "copy of it and this memory's last chunk embeds JSON rather than prose."
        )
    return result


# --------------------------------------------------------------- supersession


def resolve_ref(con: sqlite3.Connection, ref: str) -> sqlite3.Row | None:
    """A memory by id prefix, or a hot section by `path#heading`.

    Two forms because the two tiers have different notions of a stable name. A cold id
    is permanent. A hot id is `uuid5(path|heading|sha256(body))`, so it changes the next
    time anyone edits that section -- storing one would leave a reference that quietly
    stops resolving. `path#heading` survives body edits, which is the common case, and
    is the same key `eval_paraphrase.resolve_hot` uses for the fixtures.
    """
    ref = (ref or "").strip()
    if not ref:
        return None
    if "#" in ref:
        path, _, heading = ref.partition("#")
        rows = con.execute(
            "SELECT * FROM memories WHERE tier = 'hot' AND path = ? AND topic = ?",
            (str(Path(path).expanduser()) if path.startswith("~") else path,
             heading.strip()[:120]),
        ).fetchall()
        return rows[0] if len(rows) == 1 else None
    rows = con.execute(
        "SELECT * FROM memories WHERE substr(id, 1, ?) = ?", (len(ref), ref)
    ).fetchall()
    return rows[0] if len(rows) == 1 else None


def supersede(con: sqlite3.Connection, memory_id: str, by: str) -> dict:
    """Mark `memory_id` as corrected by `by`, without touching ranking or content.

    The archive cannot be edited -- that is what makes this necessary. A correction
    arrives as a *new* memory, both versions then answer the same query, and the reader
    has to work out which is current. Measured on the hot fixture: one confirmed case
    where the stale cold copy took the top slot and the corrected hot section came
    fourth, and the workaround in use is prose ("정정 — 이 항목이 우선한다") that only
    helps if the reader happens to retrieve the newer memory too.

    A hot row cannot be superseded. Its id is a hash of its body, so the mark would
    vanish the next time that file is edited, and the answer there is to edit the file:
    hot is the source of truth, not an archive of one.
    """
    row = resolve_ref(con, memory_id)
    if row is None:
        return {"status": "error", "message": f"no single memory matches {memory_id!r}"}
    if row["tier"] == "hot":
        return {
            "status": "error",
            "message": "a hot section is a file, not an archived copy — edit the file "
                       f"instead: {row['path']}",
        }
    target = resolve_ref(con, by)
    if target is None:
        return {
            "status": "error",
            "message": f"no single memory matches {by!r}. Use an id prefix, or "
                       "'path#heading' for a hot section",
        }
    if target["id"] == row["id"]:
        return {"status": "error", "message": "a memory cannot supersede itself"}
    con.execute(
        "UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE id = ?",
        (by.strip(), _now(), row["id"]),
    )
    con.commit()
    return {
        "status": "ok",
        "superseded": {"id": row["id"], "topic": row["topic"]},
        "by": {"id": target["id"], "topic": target["topic"], "tier": target["tier"]},
    }


def _supersession(con: sqlite3.Connection, row: sqlite3.Row) -> dict | None:
    """The `superseded_by` block on a recall result, resolved where it can be."""
    ref = row["superseded_by"] if "superseded_by" in row.keys() else None
    if not ref:
        return None
    block: dict[str, Any] = {"ref": ref, "at": row["superseded_at"]}
    target = resolve_ref(con, ref)
    if target is None:
        # Said out loud rather than dropped: a reference that stopped resolving is
        # itself information -- most likely the heading of a hot section changed.
        block["resolved"] = False
        return block
    block.update({"id": target["id"], "topic": target["topic"], "tier": target["tier"]})
    if target["tier"] == "hot":
        block["path"] = target["path"]
    return block


# -------------------------------------------------------------------------- recall


def _filters(
    session: str | None, project: str | None, tier: str | None
) -> tuple[str, list]:
    clauses, params = [], []
    if session:
        clauses.append("m.session = ?")
        params.append(session)
    if project:
        clauses.append("m.project = ?")
        params.append(project)
    if tier:
        clauses.append("m.tier = ?")
        params.append(tier)
    return (" AND ".join(clauses), params)


def _lexical(con, query: str, where: str, params: list, pool: int) -> list[str]:
    match = korean.match_query(query)
    if not match:
        return []
    sql = (
        "SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.memory_id "
        "WHERE f.memories_fts MATCH ?"
    )
    if where:
        sql += f" AND {where}"
    # Weights are positional per fts5: (topic, body).
    sql += (
        f" ORDER BY bm25(memories_fts, {config.TOPIC_WEIGHT}, {config.BODY_WEIGHT}) "
        "LIMIT ?"
    )
    try:
        rows = con.execute(sql, [match, *params, pool]).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r["id"] for r in rows]


def _semantic(con, query: str, where: str, params: list, pool: int) -> list[str]:
    sql = (
        "SELECT c.memory_id, c.embedding FROM memory_chunks c "
        "JOIN memories m ON m.id = c.memory_id"
    )
    if where:
        sql += f" WHERE {where}"
    rows = con.execute(sql, params).fetchall()
    if not rows:
        return []

    matrix = np.stack([embedding.from_blob(r["embedding"]) for r in rows])
    sims = matrix @ embedding.encode_query(query)

    # Max-pool chunk scores into one score per memory: a long memory should not be
    # penalised for having many chunks that do not match, only rewarded for its
    # single best-matching passage.
    best: dict[str, float] = {}
    for row, sim in zip(rows, sims):
        mid = row["memory_id"]
        if sim > best.get(mid, -1.0):
            best[mid] = float(sim)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    return [mid for mid, _ in ranked[:pool]]


def _graph_lane(con, seeds: list[str], where: str, params: list, pool: int) -> list[str]:
    """Co-citation over the concept graph: memories sharing concepts with the best hits.

    Pseudo-relevance feedback, so it runs after the base lanes and takes their top few
    as seeds. The seeds are kept rather than excluded: dropping them is the convention
    for query *expansion* but wrong for a fused lane, because it removes the memory the
    base lanes already ranked first and hands its slot to a neighbour (measured: R@1
    1.4%). Rarer entities count for more -- a concept cited by forty memories separates
    nothing.
    """
    if not seeds:
        return []
    entities = [
        r["entity"]
        for r in con.execute(
            "SELECT DISTINCT entity FROM memory_entities "
            f"WHERE memory_id IN ({','.join('?' * len(seeds))})",
            seeds,
        )
    ]
    if not entities:
        return []

    sql = (
        "SELECT me.entity, me.memory_id FROM memory_entities me "
        "JOIN memories m ON m.id = me.memory_id "
        f"WHERE me.entity IN ({','.join('?' * len(entities))})"
    )
    if where:
        sql += f" AND {where}"
    rows = con.execute(sql, [*entities, *params]).fetchall()

    cited: dict[str, int] = {}
    for r in rows:
        cited[r["entity"]] = cited.get(r["entity"], 0) + 1
    weights: dict[str, float] = {}
    # Accumulate in sorted order and tie-break on id: float addition is not
    # associative, so hash-order iteration makes ties move between processes.
    for r in sorted(rows, key=lambda r: (r["entity"], r["memory_id"])):
        mid = r["memory_id"]
        weights[mid] = weights.get(mid, 0.0) + 1.0 / cited[r["entity"]]
    return [m for m, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:pool]]


def _fuse(
    lanes: dict[str, list[str]], k: int, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """Reciprocal rank fusion.

    Rank-based rather than score-based on purpose: BM25 scores and cosine
    similarities live on different, uncalibrated scales, so any weighted sum of
    the two needs a tuning constant per corpus. RRF needs none.

    `weights` scales a lane's contribution, because plain RRF trusts every lane
    equally and the graph lane is markedly noisier than the other two.
    """
    scores: dict[str, float] = {}
    for name, ids in sorted(lanes.items()):
        w = (weights or {}).get(name, 1.0)
        for rank, mid in enumerate(ids, start=1):
            scores[mid] = scores.get(mid, 0.0) + w / (k + rank)
    return scores


def _reserve_tier_slots(
    ranked: list[tuple[str, float]], rows: dict, limit: int, reserve: int
) -> list[tuple[str, float]]:
    """Keep relevance order, but do not let either tier vanish from the response.

    Only the tail of the returned window is touched: if the top `limit` entries are
    short of `reserve` rows from a tier that has candidates, that tier's best unseen
    entries displace the weakest entries of the over-represented one. Ordering within
    the window stays by score, so this never promotes a worse match over a better one
    at the top -- it only reallocates the last slot or two.
    """
    if reserve <= 0 or limit <= 0:
        return ranked

    window, rest = ranked[:limit], ranked[limit:]
    for tier_name in ("cold", "hot"):
        present = sum(1 for mid, _ in window if _tier_of(rows, mid) == tier_name)
        waiting = [kv for kv in rest if _tier_of(rows, kv[0]) == tier_name]
        need = min(reserve, present + len(waiting)) - present
        if need <= 0:
            continue
        other = [kv for kv in window if _tier_of(rows, kv[0]) != tier_name]
        drop = {id(kv) for kv in other[-need:]}
        # Appended, not re-sorted by score. Promoted rows come from beyond the window
        # so their scores are already the lowest in it -- sorting put them here anyway.
        # Sorting also made this function depend on scores being one comparable scale,
        # which reranking breaks: it orders the head by a cross-encoder, and re-sorting
        # by fusion score threw that order away (measured: eval_recall R@5 99.2% -> 49.2%).
        window = [kv for kv in window if id(kv) not in drop] + waiting[:need]
        rest = [kv for kv in rest if kv not in waiting[:need]]
    return window + rest


def _rerank(query: str, ranked: list, rows: dict, depth: int) -> tuple[list, bool]:
    """Reorder the top `depth` by cross-encoder score, leaving the tail alone.

    Replaces the fused order rather than blending with it -- measured: blending the
    Korean reranker in as an RRF lane is strictly worse than letting it decide
    (w=0.5: R@1 79.2% vs 91.7% when it replaces). The multilingual base needed a
    blend; this one does not, which also means there is no weight to tune.

    A missing export is not fatal: the caller asked for better ranking, not for the
    request to fail, so a base-ordered answer beats no answer.
    """
    if depth <= 0 or len(ranked) < 2 or not rerank.available():
        return ranked, False
    head = ranked[:depth]
    logits = rerank.scores(
        query, [f"{rows[mid]['topic']}\n{rows[mid]['content']}" for mid, _ in head]
    )
    # Only the order changes; every row keeps its fusion score. Substituting the logit
    # was tried and is wrong: the score becomes two scales in one response, and any
    # downstream step that compares scores then mixes them (it did -- see
    # _reserve_tier_slots). `reranked` in the response is what tells a caller that the
    # head is ordered by something other than `score`.
    order = sorted(range(len(head)), key=lambda i: (-logits[i], head[i][0]))
    return [head[i] for i in order] + ranked[depth:], True


def _rescue_lane_leaders(
    ranked: list[tuple[str, float]], lanes: dict[str, list[str]], limit: int
) -> tuple[list[tuple[str, float]], int]:
    """Pull a base lane's first choice into the window when no other lane found it.

    Only the two base lanes vouch here. The graph lane keeps its seeds, so its first hit
    is usually the fused first hit already, and unaided it ranks 26% -- it is a
    tie-breaker, not a witness.

    Uncorroborated on purpose: a memory both lanes retrieved has already had its say in
    the fusion, twice. This is for the case the fusion structurally underweights -- one
    lane certain, the other never having seen it.

    Like `_reserve_tier_slots` this puts the promoted row at the *end* of the window
    rather than re-sorting, so nothing better is demoted below anything worse, and the
    fused score stays on the row. Ordering therefore stops matching `score`, which is
    already true after reranking; `rescued` in the response is what says so.
    """
    if limit <= 1 or len(ranked) <= limit:
        return ranked, 0
    rescued = 0
    for name in ("lexical", "semantic"):
        ids = lanes.get(name) or []
        if not ids:
            continue
        leader = ids[0]
        if sum(leader in found for found in lanes.values()) > 1:
            continue
        position = next((i for i, (mid, _) in enumerate(ranked) if mid == leader), None)
        if position is None or position < limit:
            continue
        # Each rescue gets its own slot. Both used to be inserted at `limit - 1`, which
        # pushed the first one straight back out: `rescued` reported 2 and the caller saw
        # one -- measured on the hot fixture, where the gold section a lane was certain
        # about ended at rank 6 because the other lane's leader took the same slot. So a
        # second rescue displaces the weakest *unrescued* row instead.
        slot = limit - 1 - rescued
        if slot < 1:
            break
        row = ranked[position]
        ranked = (ranked[:slot] + ranked[slot + 1 : limit] + [row, ranked[slot]]
                  + ranked[limit:position] + ranked[position + 1 :])
        rescued += 1
    return ranked, rescued


def _corroborated(lanes: dict[str, list[str]], memory_id: str) -> bool:
    """Do all three lanes hold this memory?

    The gate on the reranker. Measured on 170 queries (2026-08-29): where some lane
    missed the top hit -- 31 of 170 -- reranking improved 14 and hurt 2, and where all
    three held it, it improved 30 and hurt 42. So corroboration separates a 7:1 win
    from a net loss, which is the whole basis for spending 3 s on one and not the other.

    Membership in the lane's candidate list, not that lane's own first place: a lane
    missing the top hit from a pool of `CANDIDATE_POOL` means that retrieval mode never
    saw it, which is a much stronger signal than ranking it second.

    Three named lanes rather than "every lane that ran", which is what this was first
    written as and is wrong. A hybrid recall drops the graph lane entirely when the
    seeds carry no entities, and `all()` over the two survivors then reads as
    corroboration -- on the hot fixture that skipped the reranker for three queries it
    was measured to help, taking hot R@1 from 36.8% to 28.9%. An absent graph lane is a
    fact about the query, not a licence to trust the two that are left. The same
    arithmetic makes a single-lane mode ungated, which is what we want: one lane cannot
    corroborate itself.
    """
    return sum(memory_id in ids for ids in lanes.values()) >= 3


def _tier_of(rows: dict, memory_id: str) -> str:
    row = rows.get(memory_id)
    return row["tier"] if row is not None else "cold"


def _recency(created_at: str, half_life_days: int) -> float:
    if half_life_days <= 0:
        return 1.0
    try:
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        return 1.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    return 0.5 ** (age_days / half_life_days)


def recall(
    con: sqlite3.Connection,
    query: str,
    limit: int | None = None,
    session: str | None = None,
    project: str | None = None,
    detail: int = 1,
    mode: str = "hybrid",
    tier: str | None = None,
    rerank: bool | None = None,
) -> dict:
    query = query.strip()
    if not query:
        return {"status": "error", "message": "query is empty"}
    limit = limit or config.RECALL_LIMIT

    warning = db.embedding_mismatch(con) if mode != "lexical" else None
    if warning:
        mode = "lexical"

    if config.HOT_ENABLED and tier != "cold":
        hot.sync(con)

    where, params = _filters(session, project, tier)
    pool = config.CANDIDATE_POOL
    lanes: dict[str, list[str]] = {}
    if mode in ("hybrid", "lexical"):
        lanes["lexical"] = _lexical(con, query, where, params, pool)
    if mode in ("hybrid", "semantic"):
        lanes["semantic"] = _semantic(con, query, where, params, pool)

    weights: dict[str, float] = {}
    if mode == "hybrid" and config.GRAPH_LANE_WEIGHT > 0:
        seeds = [
            mid
            for mid, _ in sorted(
                _fuse(lanes, config.RRF_K).items(), key=lambda kv: (-kv[1], kv[0])
            )[: config.GRAPH_LANE_SEEDS]
        ]
        graph_ids = _graph_lane(con, seeds, where, params, pool)
        if graph_ids:
            lanes["graph"] = graph_ids
            weights["graph"] = config.GRAPH_LANE_WEIGHT

    scores = _fuse(lanes, config.RRF_K, weights)
    if not scores:
        response = {"status": "ok", "results": [], "results_count": 0, "mode": mode}
        if warning:
            response["warning"] = warning
        return response

    rows = {
        r["id"]: r
        for r in con.execute(
            "SELECT id, session, project, topic, content, created_at, tier, path, "
            "superseded_by, superseded_at "
            f"FROM memories WHERE id IN ({','.join('?' * len(scores))})",
            list(scores),
        )
    }

    def final_score(memory_id: str, fused: float) -> float:
        row = rows.get(memory_id)
        if row is None:
            return 0.0
        return fused * _recency(row["created_at"], config.HALF_LIFE_DAYS)

    ranked = sorted(
        (kv for kv in scores.items() if kv[0] in rows),
        # Tie-break on id so equal scores do not order by dict insertion, which
        # depends on which lane happened to see the memory first.
        key=lambda kv: (-final_score(*kv), kv[0]),
    )

    reranked = gated = False
    # `rerank` here is the caller's flag, which shadows the module of the same name;
    # `_rerank` below is where the module is used.
    if config.RERANK_ENABLED if rerank is None else rerank:
        # The gate overrides the default only. `rerank=True` is a caller who decided
        # this question is worth the wait, and that decision outranks the heuristic.
        gated = (
            rerank is None
            and config.RERANK_GATE
            and bool(ranked)
            and _corroborated(lanes, ranked[0][0])
        )
        if not gated:
            ranked, reranked = _rerank(query, ranked, rows, config.RERANK_DEPTH)

    # After reranking, not before: the slots exist so neither tier disappears from
    # what the caller sees, and the reranked order is what they see.
    ordered = (
        _reserve_tier_slots(ranked, rows, limit, config.MIN_TIER_SLOTS)
        if tier is None
        else ranked
    )
    # Last, so tier reservation cannot spend the rescued slot: it displaces the weakest
    # entries of the window, which is exactly where a rescue lands.
    rescued = 0
    if config.RESCUE_LANE_LEADERS:
        ordered, rescued = _rescue_lane_leaders(ordered, lanes, limit)

    cap = _DETAIL_CHARS.get(detail)
    results = []
    for mid, score in ordered[:limit]:
        row = rows.get(mid)
        if row is None:
            continue
        content = row["content"]
        entry = {
            "id": mid,
            "tier": row["tier"],
            "session": row["session"],
            "project": row["project"],
            "topic": row["topic"],
            "created_at": row["created_at"],
            "score": round(score, 5),
            "lanes": [name for name, ids in lanes.items() if mid in ids],
            "content": content if cap is None else content[:cap],
            "truncated": cap is not None and len(content) > cap,
        }
        if row["tier"] == "hot":
            entry["path"] = row["path"]
        superseded = _supersession(con, row)
        if superseded:
            entry["superseded_by"] = superseded
        results.append(entry)

    response = {
        "status": "ok",
        "mode": mode,
        "results": results,
        "results_count": len(results),
    }
    if reranked:
        response["reranked"] = config.RERANK_DEPTH
    if gated:
        # Said out loud: without it, a caller who enabled reranking cannot tell a
        # skipped rerank from a broken one, and the two want opposite responses.
        response["rerank_skipped"] = "every lane retrieved the top hit"
    if rescued:
        response["rescued"] = rescued
    if warning:
        response["warning"] = warning
    return response


def sessions_tier_counts(con: sqlite3.Connection) -> dict:
    return {
        row["tier"]: row["n"]
        for row in con.execute("SELECT tier, COUNT(*) AS n FROM memories GROUP BY tier")
    }


# ------------------------------------------------------------------------ sessions


def sessions(con: sqlite3.Connection, session: str | None = None) -> dict:
    if session is None:
        # Cold only: hot rows are sections of files, so listing them as sessions
        # would bury the archive's sessions under one entry per markdown heading.
        rows = con.execute(
            "SELECT session, COUNT(*) AS n, MAX(created_at) AS last "
            "FROM memories WHERE tier = 'cold' GROUP BY session ORDER BY last DESC"
        ).fetchall()
        hot_rows = con.execute(
            "SELECT path, COUNT(*) AS n FROM memories WHERE tier = 'hot' "
            "GROUP BY path ORDER BY path"
        ).fetchall()
        return {
            "status": "ok",
            "active_session": active_session(con),
            "sessions": [
                {"session": r["session"], "count": r["n"], "last_write": r["last"]}
                for r in rows
            ],
            "total_sessions": len(rows),
            "hot_files": len(hot_rows),
            "hot_sections": sum(r["n"] for r in hot_rows),
        }

    rows = con.execute(
        "SELECT id, topic, created_at, project, tier FROM memories "
        "WHERE session = ? ORDER BY created_at DESC",
        (session,),
    ).fetchall()
    return {
        "status": "ok",
        "session": session,
        "entries": [dict(r) for r in rows],
        "total_entries": len(rows),
    }
