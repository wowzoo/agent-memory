"""STDIO MCP server.

Surface is deliberately five tools. Measured across 76 real Claude Code sessions,
MARM's 25 tools saw 129 calls, distributed as: log_entry 99, smart_recall 11,
log_show 10, concept_build 4, concept_recall 3, graph_index 2 — and zero for
notebook, summary, delete, compaction and the four code-graph tools. The five
kept here cover all 129 of those calls (mem_graph absorbs both concept_recall and
graph_index); what was dropped is the 19 tools nobody called.
"""

from typing import Annotated, Any, Literal, NotRequired, Optional, TypedDict

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from . import db, graph, store

mcp = MCPServer("agent-memory")


def _learn_client(ctx: Context | None) -> None:
    """Hand the MCP client's own name to the store, so writes can be attributed.

    Optional in both directions: `clientInfo` is optional per spec (2026-07-28 made it
    so), and `AGENT_MEMORY_AGENT` overrides it anyway. A harness that sends nothing and
    sets nothing writes memories with `agent: null`, which is visible in the mem_write
    response rather than silently guessed at.
    """
    # `ctx.session` raises outside a request, and `MCPServer.call_tool` builds a Context
    # with no request context for in-process calls -- which made mem_write fail outright.
    # `protocol_version` is the None-safe probe for that: knowing who is writing must
    # never be the reason a write does not happen.
    if ctx is None or ctx.protocol_version is None:
        return
    params = ctx.session.client_params
    store.set_client(params.client_info.name if params else None)


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
    supersedes: Optional[str] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Store a durable memory. Korean text is fully supported.

    Write decisions, rationale, outcomes and canonical references — things worth
    recovering weeks later. Skip transient chatter.

    Parameters:
    - content: the memory text (Korean, English or mixed), at most 50,000 chars
      ($AGENT_MEMORY_MAX_CONTENT_CHARS) — split anything longer
    - session: session/topic bucket to file this under. Passing it also makes it
      the active session for *this agent's* later writes — the store is shared by
      every harness registered against it, so the active session is per agent.
      Omit to use the active session.
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
    - supersedes: the memory this one corrects, as an id prefix (or
      "path#heading" for a hot section). Use it when you are writing a memory
      because an older one turned out to be wrong or out of date: the archive
      cannot be edited, so both versions answer the same question afterwards and
      the older one is often the fuller, better-ranked account. Marking it makes
      recall say `superseded_by` on that result instead of leaving the reader to
      infer which is current. Ranking is untouched — nothing is hidden or demoted.
      A hot section cannot be superseded; edit the file.

    Returns: status, memory_id, session, topic, project, agent, chunks, and `graph`
    when entities or relations were supplied. `agent` is which harness this was
    recorded under — `$AGENT_MEMORY_AGENT`, else this client's own MCP name, else
    null. It is not a parameter: nobody gets to declare themselves someone else.
    `graph` also carries `entities_given` /
    `relations_given` **when they differ from what landed** — names are matched on
    case and spacing alone, so two spellings of one name merge into a single entity
    and a blank name is skipped. Seeing the extra key means some input did not land
    one-for-one; its absence means everything did.

    Content that looks like a credential (AWS key, GitHub/Slack/API token, private
    key, `password=...`) is refused with an error and nothing is stored — once in the
    archive its bytes outlive a delete. Redact the value and write again.
    A write nearly identical to an existing cold memory (cosine ≥ 0.95) is stored,
    and `near_duplicate` plus a `warning` name that memory: mark it with
    `supersedes` if this replaces it. No warning when `supersedes` already names it.
    """
    _learn_client(ctx)
    with db.connect() as con:
        return store.write(
            con, content, session=session, topic=topic, project=project,
            entities=entities, relations=relations, supersedes=supersedes,
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

    Parameters:
    - memory_id: id from mem_write or mem_recall
    - entities, relations: same shape as mem_write

    ⚠️ This REPLACES the memory's annotations rather than adding to them, so a call
    carrying one entity leaves it with one. To add, read what is attached first
    (`mem_graph` on one of its entities, or the `annotate.py --append` path) and send
    the full set.

    Returns: status, memory_id, entities, relations
    """
    with db.connect() as con:
        return graph.attach(con, memory_id, entities, relations)


@mcp.tool()
def mem_forget(memory_id: str) -> dict[str, Any]:
    """Permanently delete one archived (cold) memory and scrub its text from disk.

    For a memory that should not exist at all — a secret, personal data, something
    written by mistake. A memory that is merely out of date should stay and be
    corrected instead: write the correction with `mem_write(supersedes=...)`, which
    keeps the history and tells readers which one is current. This cannot be undone.

    Parameters:
    - memory_id: the id from mem_write or mem_recall, or a prefix of at least 8
      characters that matches exactly one memory

    Hot sections are refused: they are mirrored from memory files and come back on
    the next recall, so the fix is to edit the file (the error names it).

    Returns: status, memory_id, topic, session, agent, created_at (what was deleted),
    removed (chunks / entities / relations counts), scrubbed, superseded_refs.
    `scrubbed: false` comes with a `warning` — the memory is gone, but its bytes may
    remain in the WAL until the next checkpoint (another process was reading).
    `superseded_refs` lists memories this one had corrected; they keep their mark,
    which recall now shows as unresolved. Backups taken earlier are not touched.
    """
    with db.connect() as con:
        return store.forget(con, memory_id)


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
    # 50 is the lane pool (CANDIDATE_POOL): past it there is no ranking left to read,
    # only more full texts in one tool result.
    limit: Annotated[int, Field(ge=1, le=50)] = 5,
    session: Optional[str] = None,
    project: Optional[str] = None,
    detail: Literal[1, 2, 3] = 1,
    mode: Literal["hybrid", "lexical", "semantic"] = "hybrid",
    tier: Optional[Literal["hot", "cold"]] = None,
    rerank: Optional[bool] = None,
    agent: Optional[str] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Search memories by meaning and keyword. Korean queries work as well as English.

    Searches BOTH tiers at once and says which is which:
    - 'hot'  — sections of this project's Claude Code memory files. Source of truth,
               and re-indexed automatically when a file changes.
    - 'cold' — this store's archive.
    Tier is provenance, not rank: relevance alone decides the order, because boosting
    hot was measured and it promoted files even for questions they do not answer.
    So an archived copy *can* outrank the file that corrected it — measured, one case
    in the fixture. What tells you is `superseded_by` on the result, when someone
    marked the correction (see `mem_write(supersedes=...)`), plus `MIN_TIER_SLOTS`
    keeping both tiers present. Read `created_at` when neither is there.

    Three lanes are fused by reciprocal rank: a Kiwi-morpheme BM25 lane (so "패치"
    matches "패치를", which whitespace tokenizers miss), a Korean-tuned embedding
    lane over per-chunk vectors, and — in hybrid mode only, at half their
    weight — a concept-graph lane that adds memories sharing entities with the top
    hits, which is how an answer phrased in none of your words still surfaces.

    Parameters:
    - query: natural-language question or keywords, e.g. "한국어 검색이 왜 안 됐지"
    - limit: max results (default 5, at most 50)
    - session: restrict to one session; omit to search all
    - project: restrict to one project; omit to search all
    - detail: 1 = ~200 chars, 2 = ~500 chars, 3 = full content
    - mode: 'hybrid' (default) | 'lexical' (BM25 only) | 'semantic' (vectors only)
    - tier: 'hot' or 'cold' to restrict; omit to search both
    - rerank: whether to rescore the top 10 with a Korean cross-encoder. Left
      unset it is decided per query: skipped when all three lanes already
      retrieved the top hit (about five recalls in six, and `rerank_skipped`
      says so), spent otherwise. That gate is what makes it affordable — mean
      cost ~0.5 s instead of 3 s — and it is why reranking is on at all: applied
      to every query it measured *worse* than not reranking.
      True forces it past the gate, which is worth doing for "what happened with
      X" or when a first search came back vague; measured over both tiers, R@1
      55.6%→66.7% on paraphrase queries and 23.7%→71.1% when the answer is in a
      memory file. False turns it off, which is right when hunting a phrase from
      deep inside a long memory: each candidate is judged on its first ~512
      tokens, so evidence past that is invisible — on verbatim mid-document
      queries reranking takes R@1 from 82% to 33%.
    - agent: restrict to memories recorded by one harness ('claude', 'codex',
      'kiro'); omit to search every agent's. For a hot result this is whose memory
      directory the file lives in, not whoever last mirrored it. Note that memories
      written before this store recorded writers have `agent: null` and so are
      excluded by any value — use it to answer "what did Codex log", not to survey.

    Returns: status, mode, ranked results (each with tier, agent, score, the lanes
    that found it, and `path` for hot results), results_count. A `warning` field appears
    if the stored vectors were built by a different embedding model, in which case
    only the lexical lane runs. With reranking, `reranked` gives the depth rescored —
    read it as "the top N are ordered by the cross-encoder, not by `score`", since
    every row keeps its fusion score.
    """
    # Before the recall, not just before a write: recall triggers `hot.sync`, and which
    # agent owns a newly seen hot directory is recorded there.
    _learn_client(ctx)
    with db.connect() as con:
        return store.recall(
            con, query, limit=limit, session=session, project=project,
            detail=detail, mode=mode, tier=tier, rerank=rerank, agent=agent,
        )


@mcp.tool()
def mem_sessions(
    session: Optional[str] = None, ctx: Context | None = None
) -> dict[str, Any]:
    """List sessions, or list the entries inside one session.

    Sessions themselves are shared — every harness's writes are listed — but
    `active_session` is this agent's own, since that is what decides where *your*
    next unnamed write lands.

    Parameters:
    - session: omit to get every session with its entry count and last write time;
      pass a name to list that session's entries, newest first

    Returns (omitted): status, active_session, agent, sessions, total_sessions
    Returns (given):   status, session, entries (each with `agent`), total_entries
    """
    _learn_client(ctx)
    with db.connect() as con:
        return store.sessions(con, session)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
