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


def test_off_by_default(store):
    db, store_module, _, _ = store
    con = seed(db, store_module)
    assert "reranked" not in store_module.recall(con, "한국어 검색")
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
