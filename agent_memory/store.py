"""Write and recall — the two operations that carried 86% of real MARM usage."""

import os
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np

from . import config, db, embedding, graph, hot, korean

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


def write(
    con: sqlite3.Connection,
    content: str,
    session: str | None = None,
    topic: str | None = None,
    project: str | None = None,
    entities: list | None = None,
    relations: list | None = None,
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
    return result


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


def _fuse(lanes: dict[str, list[str]], k: int) -> dict[str, float]:
    """Reciprocal rank fusion.

    Rank-based rather than score-based on purpose: BM25 scores and cosine
    similarities live on different, uncalibrated scales, so any weighted sum of
    the two needs a tuning constant per corpus. RRF needs none.
    """
    scores: dict[str, float] = {}
    for ids in lanes.values():
        for rank, mid in enumerate(ids, start=1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
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
        window = [kv for kv in window if id(kv) not in drop] + waiting[:need]
        window.sort(key=lambda kv: -kv[1])
        rest = [kv for kv in rest if kv not in waiting[:need]]
    return window + rest


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

    scores = _fuse(lanes, config.RRF_K)
    if not scores:
        response = {"status": "ok", "results": [], "results_count": 0, "mode": mode}
        if warning:
            response["warning"] = warning
        return response

    rows = {
        r["id"]: r
        for r in con.execute(
            "SELECT id, session, project, topic, content, created_at, tier, path "
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
        key=lambda kv: -final_score(*kv),
    )
    ordered = (
        _reserve_tier_slots(ranked, rows, limit, config.MIN_TIER_SLOTS)
        if tier is None
        else ranked
    )

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
        results.append(entry)

    response = {
        "status": "ok",
        "mode": mode,
        "results": results,
        "results_count": len(results),
    }
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
