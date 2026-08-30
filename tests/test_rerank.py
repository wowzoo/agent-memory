"""Reranking is opt-in, replaces the fused head, and never fails the request."""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(tmp_path / "no-hot"))
    # Point the reranker at an empty directory: these tests are about the wiring, and
    # loading half a gigabyte to assert a flag is read would make them useless as a
    # fast check. The one test that needs a real model skips itself without one.
    monkeypatch.setenv("AGENT_MEMORY_RERANK_MODEL_DIR", str(tmp_path / "no-reranker"))
    from agent_memory import config, db, hot, rerank, store as store_module

    for module in (config, db, hot, rerank, store_module):
        importlib.reload(module)
    return db, store_module, config, rerank


def seed(db, store_module):
    con = db.connect()
    store_module.write(con, "한국어 검색이 안 되던 원인은 영어 전용 임베딩이었다.")
    store_module.write(con, "슬라이드 애니메이션 spid 순서를 도형 생성 순서와 맞췄다.")
    store_module.write(con, "평가셋을 20건에서 72건으로 늘렸다.")
    return con


def test_the_shipped_defaults_are_on(store, monkeypatch):
    """Guards the 2026-08-30 flip. Both were off, and the gate is what changed the
    arithmetic; a shell exporting either variable would otherwise hide the regression."""
    import importlib

    for var in ("AGENT_MEMORY_RERANK", "AGENT_MEMORY_RESCUE_LANE_LEADERS",
                "AGENT_MEMORY_RERANK_GATE"):
        monkeypatch.delenv(var, raising=False)
    from agent_memory import config as config_module

    importlib.reload(config_module)
    assert config_module.RERANK_ENABLED
    assert config_module.RERANK_GATE
    assert config_module.RESCUE_LANE_LEADERS


def test_a_skipped_rerank_is_not_a_disabled_one(store, monkeypatch):
    """With reranking on by default, an absent `reranked` key means the gate skipped it.
    Worth pinning separately: this fixture points the reranker at an empty directory, so
    a test that only checked for the absent key would pass for the wrong reason."""
    db, store_module, config, _ = store
    con = seed_with_graph(db, store_module)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))
    response = store_module.recall(con, "한국어 검색이 안 되던 원인")
    assert "reranked" not in response
    assert response["rerank_skipped"]
    con.close()


def test_a_missing_export_does_not_fail_the_request(store):
    """The caller asked for better ranking, not for an exception: a base-ordered
    answer beats no answer, so the flag degrades to a no-op."""
    db, store_module, _, rerank = store
    assert not rerank.available()
    con = seed(db, store_module)
    response = store_module.recall(con, "한국어 검색", rerank=True)
    assert response["status"] == "ok"
    assert response["results"], "results must survive a missing reranker"
    assert "reranked" not in response
    con.close()


def test_env_default_is_read_when_the_call_does_not_say(store, monkeypatch):
    db, store_module, config, _ = store
    con = seed(db, store_module)
    # rerank=False must win over the environment; None means "use the default".
    # The gate is off here on purpose: this test is about which of the two says
    # whether to rerank, and the gate is a second question asked after that one.
    monkeypatch.setattr(config, "RERANK_GATE", False)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))
    assert store_module.recall(con, "한국어 검색")["reranked"] == config.RERANK_DEPTH
    assert "reranked" not in store_module.recall(con, "한국어 검색", rerank=False)
    con.close()


def test_rerank_reorders_the_head_and_keeps_every_score(store, monkeypatch):
    """Only the order changes. Scores stay on one scale, because a downstream step
    compares them -- see test_reserved_slots_do_not_outrank_the_reranked_head."""
    db, store_module, _, rerank = store
    con = seed(db, store_module)
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    rows = {
        mid: {"topic": f"주제 {mid}", "content": f"본문 {mid}"} for mid, _ in ranked
    }
    monkeypatch.setattr(rerank, "available", lambda: True)
    monkeypatch.setattr(rerank, "scores", lambda query, passages: [-5.0, -1.0])
    out, did = store_module._rerank("질의", ranked, rows, depth=2)
    assert did
    assert out == [("b", 0.8), ("a", 0.9), ("c", 0.7)]
    con.close()


def test_reserved_slots_do_not_outrank_the_reranked_head(store):
    """The regression this cost: tier slots used to re-sort the window by score, so a
    row promoted from beyond the window jumped the whole reranked head. Two tiers only
    -- cold-only skips slot reservation, which is why it measured fine while
    eval_recall's both-tier R@5 fell from 99.2% to 49.2%."""
    _, store_module, config, _ = store
    rows = {
        "h1": {"tier": "hot"}, "h2": {"tier": "hot"}, "h3": {"tier": "hot"},
        "c1": {"tier": "cold"}, "c2": {"tier": "cold"},
    }
    # A reranked head: the order is deliberate and the scores are not descending, so a
    # row past `limit` can hold a higher fusion score than the one ranked first. That
    # is only possible once something has reordered by another signal.
    ranked = [("h1", 0.10), ("h2", 0.30), ("h3", 0.20), ("c1", 0.40), ("c2", 0.05)]
    window = store_module._reserve_tier_slots(ranked, rows, limit=3, reserve=2)
    assert [mid for mid, _ in window[:3]] == ["h1", "c1", "c2"], window


def seed_with_graph(db, store_module):
    """The same memories, annotated so the graph lane actually runs.

    Without entities a hybrid recall has two lanes, and two lanes never corroborate --
    so a gate test seeded by `seed()` would pass for the wrong reason.
    """
    con = seed(db, store_module)
    for row in con.execute("SELECT id FROM memories").fetchall():
        store_module.graph.attach(
            con, row["id"],
            entities=[{"name": "회수 품질", "type": "concept"}],
            relations=[{"source": "회수 품질", "predicate": "applies_to",
                        "target": "평가셋"}],
        )
    return con


def test_the_gate_skips_the_reranker_when_all_three_lanes_hold_the_top_hit(
    store, monkeypatch
):
    """Corroboration is read off the lanes, and the response says the skip happened."""
    db, store_module, config, _ = store
    con = seed_with_graph(db, store_module)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))

    response = store_module.recall(con, "한국어 검색이 안 되던 원인")
    assert sorted(response["results"][0]["lanes"]) == ["graph", "lexical", "semantic"]
    assert "reranked" not in response
    assert response["rerank_skipped"]
    con.close()


def test_a_missing_graph_lane_is_not_corroboration(store, monkeypatch):
    """Two lanes agreeing is not the measured signal. Read as one, the gate skipped
    three hot queries the reranker was measured to fix (R@1 36.8% -> 28.9%)."""
    db, store_module, config, _ = store
    con = seed(db, store_module)  # no entities -> no graph lane
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))

    response = store_module.recall(con, "한국어 검색이 안 되던 원인")
    assert "graph" not in response["results"][0]["lanes"]
    assert response["reranked"] == config.RERANK_DEPTH
    assert "rerank_skipped" not in response
    con.close()


def test_the_gate_yields_to_a_caller_who_asked(store, monkeypatch):
    """`rerank=True` is a decision about this question, not a default to second-guess."""
    db, store_module, config, _ = store
    con = seed(db, store_module)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))
    response = store_module.recall(con, "한국어 검색이 안 되던 원인", rerank=True)
    assert response["reranked"] == config.RERANK_DEPTH
    assert "rerank_skipped" not in response
    con.close()


def test_one_lane_cannot_corroborate_itself(store, monkeypatch):
    """A single-lane mode has no second opinion to read, so it is never gated --
    otherwise the lane that retrieved the hit would vouch for its own ordering."""
    db, store_module, config, _ = store
    con = seed(db, store_module)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))
    for mode in ("lexical", "semantic"):
        response = store_module.recall(con, "한국어 검색이 안 되던 원인", mode=mode)
        assert response["reranked"] == config.RERANK_DEPTH, mode
    con.close()


def test_a_lane_that_missed_the_top_hit_opens_the_gate(store, monkeypatch):
    db, store_module, config, _ = store
    con = seed(db, store_module)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(store_module, "_rerank", lambda *a: (a[1], True))
    # The lexical lane returns nothing, which is "this retrieval mode never saw it"
    # rather than "it ranked it second" -- the case the gate exists to catch.
    monkeypatch.setattr(store_module, "_lexical", lambda *a: [])
    response = store_module.recall(con, "한국어 검색이 안 되던 원인")
    assert response["reranked"] == config.RERANK_DEPTH
    assert "rerank_skipped" not in response
    con.close()


def test_corroboration_needs_all_three_lanes(store):
    _, store_module, _, _ = store
    lanes = {"lexical": ["a", "b"], "semantic": ["b", "a"], "graph": ["a"]}
    assert store_module._corroborated(lanes, "a")
    assert not store_module._corroborated(lanes, "b")  # the graph lane never saw it
    # Two lanes holding it is not corroboration, however unanimous they look.
    assert not store_module._corroborated({"lexical": ["a"], "semantic": ["a"]}, "a")
    assert not store_module._corroborated({"semantic": ["a"]}, "a")


def test_real_reranker_orders_the_obvious_answer_first(store, real_rerank_dir):
    db, store_module, config, _ = store
    if not (real_rerank_dir / "model.onnx").exists():
        pytest.skip(f"no reranker export in {real_rerank_dir}")
    config.RERANK_MODEL_DIR = real_rerank_dir
    con = seed(db, store_module)
    results = store_module.recall(
        con, "임베딩 모델이 영어 전용이어서 한국어가 안 됐던 일", rerank=True
    )
    assert results["reranked"] == config.RERANK_DEPTH
    assert "영어 전용 임베딩" in results["results"][0]["content"]
    con.close()


def test_rescue_promotes_an_uncorroborated_lane_leader(store):
    """A lane's own first hit, retrieved by nothing else, lands in the last slot of the
    window -- not at the top, because one lane's conviction is evidence, not a verdict."""
    _, store_module, _, _ = store
    ranked = [(f"m{i}", 1.0 - i / 100) for i in range(8)]
    lanes = {"lexical": ["m6", "m0"], "semantic": ["m0", "m1"], "graph": ["m0"]}
    out, n = store_module._rescue_lane_leaders(ranked, lanes, limit=5)
    assert n == 1
    assert [mid for mid, _ in out][:5] == ["m0", "m1", "m2", "m3", "m6"]
    # Nothing else moved, and the displaced row is still there, one place lower.
    assert [mid for mid, _ in out][5:] == ["m4", "m5", "m7"]


def test_two_rescues_do_not_share_one_slot(store):
    """Each lane certain about a different memory: both reach the window.

    Inserting both at `limit - 1` put the second one on top of the first and pushed it
    out, so `rescued` counted two and the caller received one.
    """
    _, store_module, _, _ = store
    ranked = [(f"m{i}", 1.0 - i / 100) for i in range(8)]
    lanes = {"lexical": ["m6"], "semantic": ["m7"], "graph": []}
    out, n = store_module._rescue_lane_leaders(ranked, lanes, limit=5)
    assert n == 2
    assert [mid for mid, _ in out][:5] == ["m0", "m1", "m2", "m6", "m7"]
    # The two rows they displaced are still there, just below the window.
    assert [mid for mid, _ in out][5:] == ["m3", "m4", "m5"]


def test_rescue_ignores_a_leader_another_lane_also_found(store):
    """Corroborated hits already had their say in the fusion, twice."""
    _, store_module, _, _ = store
    ranked = [(f"m{i}", 1.0 - i / 100) for i in range(8)]
    lanes = {"lexical": ["m6"], "semantic": ["m0", "m6"], "graph": []}
    out, n = store_module._rescue_lane_leaders(ranked, lanes, limit=5)
    assert n == 0 and out == ranked


def test_rescue_leaves_a_leader_already_in_the_window(store):
    _, store_module, _, _ = store
    ranked = [(f"m{i}", 1.0 - i / 100) for i in range(8)]
    lanes = {"lexical": ["m3"], "semantic": ["m0"], "graph": []}
    out, n = store_module._rescue_lane_leaders(ranked, lanes, limit=5)
    assert n == 0 and out == ranked


def test_rescue_reports_itself_only_when_it_fired(store, monkeypatch):
    """Three seeded memories cannot overflow a five-row window, so nothing is rescued and
    the key stays absent -- the caller learns the order is untouched."""
    db, store_module, config, _ = store
    con = seed(db, store_module)
    assert "rescued" not in store_module.recall(con, "한국어 검색")
    monkeypatch.setattr(store_module, "_rescue_lane_leaders", lambda o, l, n: (o, 2))
    assert store_module.recall(con, "한국어 검색")["rescued"] == 2
    monkeypatch.setattr(config, "RESCUE_LANE_LEADERS", False)
    assert "rescued" not in store_module.recall(con, "한국어 검색")
    con.close()
