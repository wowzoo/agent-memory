"""STDIO MCP server.

Surface is deliberately three tools. Measured across 76 real Claude Code sessions,
MARM's 25 tools saw 129 calls, distributed as: log_entry 99, smart_recall 11,
log_show 10, concept_build 4, concept_recall 3, graph_index 2 — and zero for
notebook, summary, delete, compaction and the four code-graph tools. The three
kept here covered 94% of actual use.
"""

from typing import Annotated, Any, Literal, NotRequired, Optional, TypedDict

from mcp.server import MCPServer
from pydantic import Field

from . import db, graph, store

mcp = MCPServer("agent-memory")


# Declared as TypedDicts rather than BaseModels on purpose: pydantic validates them
# into plain dicts, so the schema gains real field shapes while `graph.attach` keeps
# receiving the dict-or-string it already normalizes.
class Entity(TypedDict):
    name: str
    type: NotRequired[str]


class Relation(TypedDict):
    source: str
    predicate: str
    target: str


@mcp.tool()
def mem_write(
    content: str,
    session: Optional[str] = None,
    topic: Optional[str] = None,
    project: Optional[str] = None,
    entities: Optional[list[Entity | str]] = None,
    relations: Optional[list[Relation]] = None,
) -> dict[str, Any]:
    """Store a durable memory. Korean text is fully supported.

    Write decisions, rationale, outcomes and canonical references — things worth
    recovering weeks later. Skip transient chatter.

    Parameters:
    - content: the memory text (Korean, English or mixed)
    - session: session/topic bucket to file this under. Passing it also makes it
      the active session for later writes. Omit to use the active session.
    - topic: short label; defaults to the first line of `content`
    - project: defaults to $AGENT_MEMORY_PROJECT, else the current directory name
    - entities: optional concept-graph nodes, e.g.
      [{"name": "Kiwi", "type": "tool"}, {"name": "조사 분리 실패", "type": "error"}]
      Types are free-form; concept / decision / error / tool / person / project are
      the conventional ones. A bare string is accepted as {"name": s}.
    - relations: optional typed edges between entities, e.g.
      [{"source": "Kiwi", "predicate": "fixes", "target": "조사 분리 실패"}]
      Endpoints not listed in `entities` are created automatically. Use the
      predicate you actually mean — this is why extraction is yours and not the
      server's: it cannot recover a predicate from Korean prose.

    Returns: status, memory_id, session, topic, project, chunks, and `graph` when
    entities or relations were supplied.
    """
    with db.connect() as con:
        return store.write(
            con, content, session=session, topic=topic, project=project,
            entities=entities, relations=relations,
        )


@mcp.tool()
def mem_annotate(
    memory_id: str,
    entities: Optional[list[Entity | str]] = None,
    relations: Optional[list[Relation]] = None,
) -> dict[str, Any]:
    """Attach concept-graph entities and relations to a memory that already exists.

    Use this to build the graph over memories written before annotation, including
    everything carried over from MARM. Read the memory first (mem_recall with
    detail=3), then annotate what it actually says.

    Replaces that memory's previous entities and relations rather than merging, so
    re-annotating is a correction and not an accumulation.

    Parameters:
    - memory_id: id from mem_write or mem_recall
    - entities, relations: same shape as mem_write

    Returns: status, memory_id, entities, relations
    """
    with db.connect() as con:
        return graph.attach(con, memory_id, entities, relations)


@mcp.tool()
def mem_graph(
    entity: Optional[str] = None,
    hops: Annotated[int, Field(ge=1, le=graph.MAX_HOPS)] = 1,
    predicate: Optional[str] = None,
) -> dict[str, Any]:
    """Traverse the concept graph, or survey its shape.

    Two modes:
    - entity given: entities and relations within `hops` of it, each relation
      carrying the memory that asserted it so you can go read the evidence
    - entity omitted: totals, hub entities by degree, isolated entities (recorded
      but never related to anything), and the predicate distribution

    Parameters:
    - entity: name to start from; matching ignores case and spacing
    - hops: traversal depth, 1–4 (default 1)
    - predicate: restrict traversal to one relation type, e.g. "fixes"

    Returns (entity given): status, entity, type, hops, nodes, relations, relations_count.
    `status: "not_found"` includes `did_you_mean` candidates.
    Returns (omitted): status, totals, hubs, isolated, predicates
    """
    with db.connect() as con:
        if entity:
            return graph.neighborhood(con, entity, hops=hops, predicate=predicate)
        return graph.overview(con)


@mcp.tool()
def mem_recall(
    query: str,
    limit: int = 5,
    session: Optional[str] = None,
    project: Optional[str] = None,
    detail: Literal[1, 2, 3] = 1,
    mode: Literal["hybrid", "lexical", "semantic"] = "hybrid",
    tier: Optional[Literal["hot", "cold"]] = None,
) -> dict[str, Any]:
    """Search memories by meaning and keyword. Korean queries work as well as English.

    Searches BOTH tiers at once and says which is which:
    - 'hot'  — sections of this project's Claude Code memory files. Source of truth,
               and re-indexed automatically when a file changes.
    - 'cold' — this store's archive.
    When both tiers hold the same fact the hot one is ranked above its archived copy,
    so a stale archive entry cannot quietly outrank the file that corrected it.

    Two lanes run and are fused by reciprocal rank: a Kiwi-morpheme BM25 lane
    (so "패치" matches "패치를", which whitespace tokenizers miss) and a
    Korean-tuned embedding lane over per-chunk vectors.

    Parameters:
    - query: natural-language question or keywords, e.g. "한국어 검색이 왜 안 됐지"
    - limit: max results (default 5)
    - session: restrict to one session; omit to search all
    - project: restrict to one project; omit to search all
    - detail: 1 = ~200 chars, 2 = ~500 chars, 3 = full content
    - mode: 'hybrid' (default) | 'lexical' (BM25 only) | 'semantic' (vectors only)
    - tier: 'hot' or 'cold' to restrict; omit to search both

    Returns: status, mode, ranked results (each with tier, score, the lanes that
    found it, and `path` for hot results), results_count. A `warning` field appears
    if the stored vectors were built by a different embedding model, in which case
    only the lexical lane runs.
    """
    with db.connect() as con:
        return store.recall(
            con, query, limit=limit, session=session, project=project,
            detail=detail, mode=mode, tier=tier,
        )


@mcp.tool()
def mem_sessions(session: Optional[str] = None) -> dict[str, Any]:
    """List sessions, or list the entries inside one session.

    Parameters:
    - session: omit to get every session with its entry count and last write time;
      pass a name to list that session's entries, newest first

    Returns (omitted): status, active_session, sessions, total_sessions
    Returns (given):   status, session, entries, total_entries
    """
    with db.connect() as con:
        return store.sessions(con, session)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
