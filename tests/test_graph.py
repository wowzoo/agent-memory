"""Concept graph: attachment, traversal, and the invariants that keep it honest."""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(tmp_path / "no-hot"))
    from agent_memory import config, db, graph, hot, store as store_module

    for module in (config, db, hot, graph, store_module):
        importlib.reload(module)
    return db, graph, store_module


def test_write_attaches_entities_and_relations(store):
    db, graph, store_module = store
    con = db.connect()
    result = store_module.write(
        con,
        "MARM 의 concept graph 는 한국어 조사를 분리하지 못해 엔티티 경계가 어긋났다.",
        entities=[{"name": "MARM concept graph", "type": "tool"},
                  {"name": "조사 분리 실패", "type": "error"}],
        relations=[{"source": "MARM concept graph", "predicate": "causes",
                    "target": "조사 분리 실패"}],
    )
    assert result["graph"] == {"entities": 2, "relations": 1}

    hood = graph.neighborhood(con, "MARM concept graph")
    assert hood["status"] == "ok"
    assert hood["relations_count"] == 1
    edge = hood["relations"][0]
    # Surface forms, not the lowercased join keys — one spelling per entity.
    assert (edge["source"], edge["predicate"], edge["target"]) == (
        "MARM concept graph", "causes", "조사 분리 실패",
    )
    # The relation cites the memory that asserted it.
    assert edge["memory_id"] == result["memory_id"]
    con.close()


def test_relation_endpoints_are_created_without_being_declared(store):
    """An agent should be able to state an edge without also listing both nodes."""
    db, graph, store_module = store
    con = db.connect()
    store_module.write(
        con, "Kiwi 로 조사 분리 문제를 해결했다.",
        relations=[{"source": "Kiwi", "predicate": "fixes", "target": "조사 분리 실패"}],
    )
    hood = graph.neighborhood(con, "kiwi")
    assert {n["display"] for n in hood["nodes"]} == {"Kiwi", "조사 분리 실패"}
    con.close()


def test_matching_ignores_case_and_spacing(store):
    db, graph, store_module = store
    con = db.connect()
    store_module.write(con, "x", entities=[{"name": "AI-DLC  v2", "type": "project"}])
    assert graph.neighborhood(con, "ai-dlc v2")["status"] == "ok"
    assert graph.neighborhood(con, "AI-DLC V2")["status"] == "ok"
    con.close()


def test_multi_hop_traversal_and_depth_limit(store):
    db, graph, store_module = store
    con = db.connect()
    store_module.write(con, "a", relations=[
        {"source": "A", "predicate": "uses", "target": "B"},
        {"source": "B", "predicate": "uses", "target": "C"},
        {"source": "C", "predicate": "uses", "target": "D"},
    ])
    assert {n["display"] for n in graph.neighborhood(con, "A", hops=1)["nodes"]} == {"A", "B"}
    assert {n["display"] for n in graph.neighborhood(con, "A", hops=2)["nodes"]} == {"A", "B", "C"}
    assert graph.neighborhood(con, "A", hops=99)["hops"] == graph.MAX_HOPS
    con.close()


def test_predicate_filter(store):
    db, graph, store_module = store
    con = db.connect()
    store_module.write(con, "a", relations=[
        {"source": "A", "predicate": "uses", "target": "B"},
        {"source": "A", "predicate": "replaces", "target": "C"},
    ])
    hood = graph.neighborhood(con, "A", predicate="replaces")
    assert [e["target"] for e in hood["relations"]] == ["C"]
    con.close()


def test_reannotating_replaces_rather_than_accumulates(store):
    db, graph, store_module = store
    con = db.connect()
    mid = store_module.write(con, "a", entities=["A", "B"],
                             relations=[{"source": "A", "predicate": "uses",
                                         "target": "B"}])["memory_id"]
    graph.attach(con, mid, entities=["A"], relations=[])
    assert graph.overview(con)["totals"] == {
        "entities": 1, "relations": 0, "annotated_memories": 1, "total_memories": 1,
    }
    # B was referenced by nothing after the correction, so it is gone.
    assert graph.neighborhood(con, "B")["status"] == "not_found"
    con.close()


def test_deleting_a_memory_removes_its_edges(store):
    db, graph, store_module = store
    con = db.connect()
    mid = store_module.write(con, "a", relations=[
        {"source": "A", "predicate": "uses", "target": "B"}])["memory_id"]
    con.execute("DELETE FROM memories WHERE id = ?", (mid,))
    con.commit()
    assert graph.overview(con)["totals"]["relations"] == 0
    con.close()


def test_self_edges_and_blank_names_are_dropped(store):
    db, graph, store_module = store
    con = db.connect()
    result = store_module.write(con, "a", entities=["", "   "], relations=[
        {"source": "A", "predicate": "uses", "target": "A"},
        {"source": "A", "predicate": "", "target": "B"},
        {"source": "", "predicate": "uses", "target": "B"},
    ])
    assert result["graph"] == {"entities": 0, "relations": 0}
    con.close()


def test_overview_reports_hubs_and_isolated(store):
    db, graph, store_module = store
    con = db.connect()
    store_module.write(con, "a", entities=[{"name": "혼자", "type": "concept"}],
                       relations=[{"source": "허브", "predicate": "uses", "target": "X"},
                                  {"source": "허브", "predicate": "uses", "target": "Y"},
                                  {"source": "허브", "predicate": "fixes", "target": "Z"}])
    view = graph.overview(con)
    assert view["hubs"][0]["display"] == "허브"
    assert view["hubs"][0]["degree"] == 3
    assert [r["display"] for r in view["isolated"]] == ["혼자"]
    assert {p["predicate"]: p["n"] for p in view["predicates"]} == {"uses": 2, "fixes": 1}
    con.close()


def test_annotate_rejects_unknown_memory(store):
    db, graph, _ = store
    con = db.connect()
    assert graph.attach(con, "nope", entities=["A"])["status"] == "error"
    con.close()


def test_graph_lane_prefers_neighbours_sharing_the_rarer_concept(store):
    """A concept on every memory separates nothing; a concept on two is evidence."""
    db, _, store_module = store
    con = db.connect()
    seed = store_module.write(
        con, "롯데온 MO 병렬개발 산출물.", entities=["롯데온 MO", "병렬개발"]
    )["memory_id"]
    rare = store_module.write(con, "덱 작업.", entities=["병렬개발"])["memory_id"]
    common = [
        store_module.write(con, f"기억 {i}.", entities=["롯데온 MO"])["memory_id"]
        for i in range(4)
    ]

    ids = store_module._graph_lane(con, [seed], "", [], 50)
    assert ids[0] == seed  # seeds are kept, not excluded
    assert ids[1] == rare
    assert set(ids[2:]) == set(common)
    con.close()


def test_graph_lane_is_off_outside_hybrid_and_when_disabled(store):
    db, _, store_module = store
    from agent_memory import config
    con = db.connect()
    store_module.write(con, "롯데온 MO 병렬개발 산출물.", entities=["롯데온 MO"])
    store_module.write(con, "롯데온 MO 분석 덱.", entities=["롯데온 MO"])

    def lanes(**kw):
        results = store_module.recall(con, "롯데온 MO", limit=5, **kw)["results"]
        return {lane for r in results for lane in r["lanes"]}

    assert "graph" in lanes()
    assert "graph" not in lanes(mode="lexical")
    config.GRAPH_LANE_WEIGHT = 0.0
    assert "graph" not in lanes()
    con.close()


def test_unannotated_memories_are_visible_in_totals(store):
    """The graph should say how much of the store it actually covers."""
    db, graph, store_module = store
    con = db.connect()
    store_module.write(con, "annotated", entities=["A"])
    store_module.write(con, "plain, no entities")
    totals = graph.overview(con)["totals"]
    assert totals["annotated_memories"] == 1
    assert totals["total_memories"] == 2
    con.close()
