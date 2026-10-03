"""`recall(trace=True)` records each stage and changes nothing about the answer."""

import importlib

import pytest

NOTES = [
    "리랭크 게이트는 세 레인이 모두 1위를 잡으면 리랭커를 건너뛴다.",
    "단독 레인 1위 구제는 한 레인만 본 1위를 창의 마지막 칸에 넣는다.",
    "tier 슬롯은 창에서 한 계층이 사라지지 않게 마지막 칸을 재배분한다.",
    "그래프 레인은 상위 결과와 엔티티를 공유하는 기억을 더한다.",
    "Kiwi 형태소 분석으로 렉시컬 레인이 조사를 떼고 찾는다.",
    "대시보드는 읽기 전용 연결로만 스토어를 연다.",
]


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(tmp_path / "no-hot"))
    monkeypatch.setenv("AGENT_MEMORY_RERANK_MODEL_DIR", str(tmp_path / "no-reranker"))
    from agent_memory import config, db, hot, rerank, store as store_module

    for module in (config, db, hot, rerank, store_module):
        importlib.reload(module)
    con = db.connect()
    for i, note in enumerate(NOTES):
        store_module.write(con, note, session="t", entities=[f"축{i % 2}"])
    yield con, store_module, rerank
    con.close()


def fake_reranker(monkeypatch, rerank):
    """Scores passages by length, so the head order is known and differs from fusion."""
    monkeypatch.setattr(rerank, "available", lambda: True)
    monkeypatch.setattr(rerank, "scores", lambda q, passages: [float(len(p)) for p in passages])


def strip(response: dict) -> dict:
    return {k: v for k, v in response.items() if k != "trace"}


@pytest.mark.parametrize("rerank_arg", [None, True, False])
def test_tracing_does_not_change_the_answer(store, monkeypatch, rerank_arg):
    con, store_module, rerank = store
    fake_reranker(monkeypatch, rerank)
    plain = store_module.recall(con, "리랭크 게이트", limit=3, rerank=rerank_arg)
    traced = store_module.recall(con, "리랭크 게이트", limit=3, rerank=rerank_arg, trace=True)
    assert "trace" not in plain
    assert strip(traced) == plain


def test_trace_records_each_stage(store, monkeypatch):
    con, store_module, rerank = store
    fake_reranker(monkeypatch, rerank)
    out = store_module.recall(con, "리랭크 게이트", limit=3, rerank=True, trace=True)
    t = out["trace"]
    assert t["final"] == [r["id"] for r in out["results"]]
    assert set(t["lanes"]) >= {"lexical", "semantic"}
    # Forced rerank: every head row has a logit, and the reranked head follows them.
    assert out["reranked"] and set(t["logits"]) == set(t["fused"][: len(t["logits"])])
    head = t["reranked"][: len(t["logits"])]
    assert [t["logits"][m] for m in head] == sorted(t["logits"].values(), reverse=True)
    assert t["before_slots"] == t["reranked"][:3]
    for mid in t["final"]:
        assert t["memories"][mid]["topic"]
    assert {"lexical", "semantic", "fuse", "rerank", "slots_rescue"} <= set(t["timings_ms"])


def test_trace_without_rerank_has_no_reranked_stage(store):
    con, store_module, _ = store
    t = store_module.recall(con, "형태소 분석", limit=3, rerank=False, trace=True)["trace"]
    assert t["reranked"] is None and t["logits"] == {}
    assert t["before_slots"] == t["fused"][:3]


def test_a_recall_that_finds_nothing_still_has_a_trace(store):
    con, store_module, _ = store
    out = store_module.recall(con, "qqqzzz", limit=3, mode="lexical", trace=True)
    assert out["results"] == []
    assert out["trace"]["final"] == [] and "lexical" in out["trace"]["timings_ms"]
