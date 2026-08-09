"""Concept graph: entities and typed relations, supplied by the writing agent.

Why the agent and not an extractor: MARM's spaCy pipeline was measured on this exact
corpus and produced 361 entities / 0 relations, with entity boundaries cut inside
Korean particles because `en_core_web_sm` cannot separate 은/는/가/를. Switching to
spaCy's Korean model does not help — it ships no `noun_chunks` (`syntax_iterators`
is empty), and noun chunks are the input to that pipeline. An LLM already holding the
memory in context knows the entities without guessing at morpheme boundaries, and it
knows the predicate rather than falling back to co-occurrence.

Traversal is BFS over the relation table. No graph engine: the whole point of the
measurement above is that the hard part was never the traversal.
"""

import re
import sqlite3
from datetime import datetime, timezone

_WS = re.compile(r"\s+")
MAX_HOPS = 4


def normalize(name: str) -> str:
    """Join key for an entity.

    Case and whitespace only. Korean particles are deliberately NOT stripped here:
    entity names arrive from the agent already in dictionary form, and guessing at
    morpheme boundaries is the failure this design exists to avoid.
    """
    return _WS.sub(" ", name).strip().lower()


def _upsert_entity(con, display: str, entity_type: str = "concept") -> str | None:
    display = _WS.sub(" ", display or "").strip()
    if not display:
        return None
    key = normalize(display)
    con.execute(
        "INSERT INTO entities(name, display, type, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        # A later explicit type wins over the 'concept' default a relation endpoint
        # was created with; it never downgrades back to the default.
        "  type = CASE WHEN excluded.type != 'concept' THEN excluded.type ELSE type END",
        (key, display, entity_type or "concept", datetime.now(timezone.utc).isoformat()),
    )
    return key


def attach(
    con: sqlite3.Connection,
    memory_id: str,
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> dict:
    """Attach entities and relations to an existing memory. Replaces prior ones."""
    row = con.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        return {"status": "error", "message": f"no such memory: {memory_id}"}

    con.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
    con.execute("DELETE FROM relations WHERE memory_id = ?", (memory_id,))

    attached: set[str] = set()
    for item in entities or []:
        # Accept either {"name": ..., "type": ...} or a bare string, because an agent
        # writing a quick note should not have to wrap every name in an object.
        name = item.get("name", "") if isinstance(item, dict) else str(item)
        etype = item.get("type", "concept") if isinstance(item, dict) else "concept"
        key = _upsert_entity(con, name, etype)
        if key:
            attached.add(key)

    kept_relations = 0
    for item in relations or []:
        if not isinstance(item, dict):
            continue
        source = _upsert_entity(con, item.get("source", ""))
        target = _upsert_entity(con, item.get("target", ""))
        predicate = _WS.sub("_", (item.get("predicate") or "").strip().lower())
        if not (source and target and predicate) or source == target:
            continue
        con.execute(
            "INSERT OR IGNORE INTO relations(source, predicate, target, memory_id) "
            "VALUES (?, ?, ?, ?)",
            (source, predicate, target, memory_id),
        )
        # Relation endpoints are attached to the memory too, so a relation always
        # has a memory to cite and never becomes a floating edge.
        attached.update({source, target})
        kept_relations += 1

    con.executemany(
        "INSERT OR IGNORE INTO memory_entities(memory_id, entity) VALUES (?, ?)",
        [(memory_id, name) for name in sorted(attached)],
    )
    _prune_unused_entities(con)
    con.commit()
    return {
        "status": "ok",
        "memory_id": memory_id,
        "entities": len(attached),
        "relations": kept_relations,
    }


def _prune_unused_entities(con) -> None:
    """Drop entities no memory references any more, so re-annotating cannot leave
    a growing pile of unreachable names behind."""
    con.execute(
        "DELETE FROM entities WHERE name NOT IN (SELECT entity FROM memory_entities)"
    )


# ------------------------------------------------------------------------ queries


def _neighbours(con, names: set[str], predicate: str | None) -> list[sqlite3.Row]:
    if not names:
        return []
    placeholders = ",".join("?" * len(names))
    # Surface forms come along for the ride: relations store the normalized join key,
    # and returning that raw would show the caller two spellings of one entity —
    # `nodes` saying "AI-DLC 2.5.11" while `relations` says "ai-dlc 2.5.11".
    sql = (
        "SELECT r.source, r.predicate, r.target, r.memory_id, m.topic, m.tier, "
        "       es.display AS source_display, et.display AS target_display "
        "FROM relations r "
        "JOIN memories m ON m.id = r.memory_id "
        "JOIN entities es ON es.name = r.source "
        "JOIN entities et ON et.name = r.target "
        f"WHERE (r.source IN ({placeholders}) OR r.target IN ({placeholders}))"
    )
    params = [*names, *names]
    if predicate:
        sql += " AND r.predicate = ?"
        params.append(predicate)
    return con.execute(sql, params).fetchall()


def neighborhood(
    con: sqlite3.Connection,
    entity: str,
    hops: int = 1,
    predicate: str | None = None,
) -> dict:
    """Entities and relations within `hops` of `entity`, with the memories citing them."""
    start = normalize(entity)
    row = con.execute(
        "SELECT name, display, type FROM entities WHERE name = ?", (start,)
    ).fetchone()
    if row is None:
        near = con.execute(
            "SELECT display FROM entities WHERE name LIKE ? LIMIT 5", (f"%{start}%",)
        ).fetchall()
        return {
            "status": "not_found",
            "entity": entity,
            "did_you_mean": [r["display"] for r in near],
        }

    hops = max(1, min(hops, MAX_HOPS))
    seen = {start}
    frontier = {start}
    edges: dict[tuple, sqlite3.Row] = {}
    for _ in range(hops):
        rows = _neighbours(con, frontier, predicate)
        next_frontier = set()
        for r in rows:
            edges[(r["source"], r["predicate"], r["target"], r["memory_id"])] = r
            for end in (r["source"], r["target"]):
                if end not in seen:
                    next_frontier.add(end)
        seen |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    names = ",".join("?" * len(seen))
    nodes = con.execute(
        f"SELECT e.name, e.display, e.type, "
        f"(SELECT COUNT(*) FROM memory_entities me WHERE me.entity = e.name) AS memories "
        f"FROM entities e WHERE e.name IN ({names}) ORDER BY memories DESC",
        list(seen),
    ).fetchall()

    return {
        "status": "ok",
        "entity": row["display"],
        "type": row["type"],
        "hops": hops,
        "nodes": [dict(n) for n in nodes],
        "relations": [
            {
                "source": r["source_display"],
                "predicate": r["predicate"],
                "target": r["target_display"],
                "memory_id": r["memory_id"],
                "memory_topic": r["topic"],
                "tier": r["tier"],
            }
            for r in edges.values()
        ],
        "relations_count": len(edges),
    }


def overview(con: sqlite3.Connection, limit: int = 10) -> dict:
    """Shape of the graph, plus the two defects worth knowing about.

    - hubs:     entities with the most relations
    - isolated: entities recorded on memories but never related to anything, i.e.
                a fact was captured and never connected
    """
    totals = {
        "entities": con.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
        "relations": con.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
        "annotated_memories": con.execute(
            "SELECT COUNT(DISTINCT memory_id) FROM memory_entities"
        ).fetchone()[0],
        "total_memories": con.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
    }

    hubs = con.execute(
        "SELECT e.display, e.type, COUNT(*) AS degree FROM entities e "
        "JOIN relations r ON r.source = e.name OR r.target = e.name "
        "GROUP BY e.name ORDER BY degree DESC, e.display LIMIT ?",
        (limit,),
    ).fetchall()

    isolated = con.execute(
        "SELECT e.display, e.type, "
        "(SELECT COUNT(*) FROM memory_entities me WHERE me.entity = e.name) AS memories "
        "FROM entities e WHERE NOT EXISTS "
        "(SELECT 1 FROM relations r WHERE r.source = e.name OR r.target = e.name) "
        "ORDER BY memories DESC, e.display LIMIT ?",
        (limit,),
    ).fetchall()

    predicates = con.execute(
        "SELECT predicate, COUNT(*) AS n FROM relations "
        "GROUP BY predicate ORDER BY n DESC"
    ).fetchall()

    return {
        "status": "ok",
        "totals": totals,
        "hubs": [dict(r) for r in hubs],
        "isolated": [dict(r) for r in isolated],
        "predicates": [dict(r) for r in predicates],
    }
