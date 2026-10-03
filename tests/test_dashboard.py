"""The dashboard reads the store and never writes it, and its checks see what they claim to."""

import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

NOTE = "# 대시보드 메모\n\n상태 탭은 읽기 전용 연결로만 스토어를 연다.\n"


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(hot_dir))
    monkeypatch.setenv("AGENT_MEMORY_RERANK_MODEL_DIR", str(tmp_path / "no-reranker"))
    from agent_memory import config, db, hot, rerank, store as store_module

    for module in (config, db, hot, rerank, store_module):
        importlib.reload(module)
    import dashboard

    importlib.reload(dashboard)
    assert config.HOT_ENABLED  # importing the dashboard must not turn sync off
    monkeypatch.setattr(config, "HOT_ENABLED", False)  # what main() does for the server
    return hot_dir, db, hot, store_module, dashboard


def check(dashboard, label):
    con = dashboard.connect_ro()
    try:
        return next(c for c in dashboard.status(con)["checks"] if c["label"] == label)
    finally:
        con.close()


def test_the_connection_refuses_writes(store):
    _, db, _, store_module, dashboard = store
    con = db.connect()
    store_module.write(con, "읽기 전용 확인용 기록.", session="t")
    con.close()
    ro = dashboard.connect_ro()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.execute("DELETE FROM memories")
    ro.close()


def test_a_missing_store_is_not_created(store, tmp_path):
    *_, dashboard = store
    with pytest.raises(sqlite3.OperationalError):
        dashboard.connect_ro()
    assert not (tmp_path / "memory.db").exists()


def test_a_clean_store_passes_every_integrity_check(store):
    hot_dir, db, hot, store_module, dashboard = store
    (hot_dir / "note.md").write_text(NOTE, encoding="utf-8")
    con = db.connect()
    hot.sync(con)
    old = store_module.write(con, "검사는 별도 스크립트로 둔다.", session="t")
    store_module.write(con, "검사는 기존 스크립트에 흡수했다.", session="t",
                       supersedes=old["memory_id"][:8])
    con.close()
    ro = dashboard.connect_ro()
    checks = dashboard.status(ro)["checks"]
    ro.close()
    assert [c["label"] for c in checks if c["level"] == "error"] == []


def test_a_memory_missing_from_fts_is_an_error(store):
    _, db, _, store_module, dashboard = store
    con = db.connect()
    written = store_module.write(con, "FTS 행이 빠지면 렉시컬 레인이 이 기억을 못 찾는다.", session="t")
    con.execute("DELETE FROM memories_fts WHERE memory_id = ?", (written["memory_id"],))
    con.commit()
    con.close()
    fts = check(dashboard, "FTS 정합성")
    assert fts["level"] == "error"
    assert "누락 1" in fts["value"]


def test_hot_sources_edited_or_gone_are_reported_not_synced(store):
    hot_dir, db, hot, _, dashboard = store
    edited, gone = hot_dir / "edited.md", hot_dir / "gone.md"
    edited.write_text(NOTE, encoding="utf-8")
    gone.write_text("# 사라질 메모\n\n이 파일은 색인 뒤에 지워진다.\n", encoding="utf-8")
    con = db.connect()
    hot.sync(con)
    con.close()
    edited.write_text(NOTE + "\n덧붙인 줄.\n", encoding="utf-8")
    gone.unlink()

    changed = check(dashboard, "hot 원본 변경")
    assert [r["topic"] for r in changed["rows"]] == ["edited.md"]
    missing = check(dashboard, "hot 원본 경로 없음")
    assert len(missing["rows"]) == 1
    assert "마지막 사본" in missing["rows"][0]["detail"]

    # Reporting must not have re-indexed anything: the gone file's row is still there.
    con = db.connect()
    assert con.execute("SELECT COUNT(*) FROM memories WHERE path = ?", (str(gone),)).fetchone()[0] == 1
    con.close()


def test_a_percent_sign_is_searched_as_text(store):
    _, db, _, store_module, dashboard = store
    con = db.connect()
    store_module.write(con, "R@5 가 84.7% 였다.", session="t")
    store_module.write(con, "퍼센트 기호가 없는 기록.", session="t")
    con.close()
    ro = dashboard.connect_ro()
    assert dashboard.memories(ro, {"q": "%"})["total"] == 1
    ro.close()


def test_detail_follows_a_correction_both_ways(store):
    _, db, _, store_module, dashboard = store
    con = db.connect()
    old = store_module.write(con, "빈 장 검사는 별도 스크립트로 신설했다.", session="t")
    new = store_module.write(con, "빈 장 검사는 기존 검사에 흡수했다.", session="t",
                             supersedes=old["memory_id"][:8])
    con.close()
    ro = dashboard.connect_ro()
    assert dashboard.memory(ro, old["memory_id"])["supersession"]["id"] == new["memory_id"]
    assert [c["id"] for c in dashboard.memory(ro, new["memory_id"])["corrects"]] == [old["memory_id"]]
    assert dashboard.memory(ro, "no-such-id") is None
    ro.close()


def build_star(db, store_module):
    """A hub with twelve spokes (a-b also linked, hub-a stated twice), a smaller hub with six,
    and a separate x-y pair in another project."""
    con = db.connect()
    spokes = "abcdefghijkl"
    store_module.write(
        con, "허브와 바퀴살 열두 개.", session="t", project="p",
        entities=[{"name": "a", "type": "error"}],
        relations=[{"source": "hub", "predicate": "has", "target": s} for s in spokes]
        + [{"source": "a", "predicate": "near", "target": "b"}])
    store_module.write(con, "작은 허브와 바퀴살 여섯 개.", session="t", project="p",
                       relations=[{"source": "hub2", "predicate": "has", "target": s}
                                  for s in "mnopqr"])
    store_module.write(con, "허브와 a 를 한 번 더 말한 기록.", session="t", project="p",
                       relations=[{"source": "a", "predicate": "part_of", "target": "hub"}])
    store_module.write(con, "다른 프로젝트의 x 와 y.", session="t", project="other",
                       relations=[{"source": "x", "predicate": "uses", "target": "y"}])
    con.close()


def test_graph_takes_hubs_with_their_best_spokes(store):
    _, db, _, store_module, dashboard = store
    build_star(db, store_module)
    ro = dashboard.connect_ro()
    g = dashboard.graph(ro, {"limit": "10"})
    names = [n["name"] for n in g["nodes"]]
    # Each hub with its four best spokes. A plain top-10 by degree would be hub, hub2, a, b,
    # c-h: hub2 with none of its spokes, an isolated dot.
    assert names == ["hub", "a", "b", "c", "d", "hub2", "m", "n", "o", "p"]
    linked = {e["source"] for e in g["edges"]} | {e["target"] for e in g["edges"]}
    assert set(names) == linked
    by_name = {n["name"]: n for n in g["nodes"]}
    # Distinct neighbours, not relation rows: hub-a is two rows under two predicates.
    assert by_name["hub"]["degree"] == 12
    assert {(e["source"], e["target"]): e["weight"] for e in g["edges"]}[("a", "hub")] == 2
    ro.close()


def test_graph_filters_by_project_and_hides_groups(store):
    _, db, _, store_module, dashboard = store
    build_star(db, store_module)
    ro = dashboard.connect_ro()
    assert {n["name"] for n in dashboard.graph(ro, {"project": "other"})["nodes"]} == {"x", "y"}
    hidden = dashboard.graph(ro, {"hide": "오류"})
    names = {n["name"] for n in hidden["nodes"]}
    assert "a" not in names
    # Still in the legend with its count, or there is nothing to click to show it again.
    assert {"name": "오류", "count": 1, "hidden": True} in hidden["groups"]
    assert {n["name"]: n["degree"] for n in hidden["nodes"]}["b"] == 1
    assert dashboard.memories(ro, {"entity": "x"})["total"] == 1
    ro.close()


def test_recall_matches_the_store_and_does_not_sync(store):
    hot_dir, db, _, store_module, dashboard = store
    con = db.connect()
    store_module.write(con, "대시보드 회수 탭은 mem_recall 과 같은 결과를 내야 한다.", session="t")
    store_module.write(con, "그래프 탭은 relation 을 그린다.", session="t")
    con.close()
    (hot_dir / "unsynced.md").write_text(NOTE, encoding="utf-8")  # on disk, never mirrored

    ro = dashboard.connect_ro()
    got = dashboard.recall(ro, {"query": "대시보드 회수 탭", "rerank": "off"})
    ro.close()
    assert got["status"] == "ok"
    con = db.connect()
    assert con.execute("SELECT COUNT(*) FROM memories WHERE tier = 'hot'").fetchone()[0] == 0
    want = store_module.recall(con, "대시보드 회수 탭", limit=5, rerank=False)
    con.close()
    assert [r["id"] for r in got["results"]] == [r["id"] for r in want["results"]]


def test_recall_log_keeps_ambiguous_apart_from_missing(store, tmp_path, monkeypatch):
    hot_dir, db, hot, store_module, dashboard = store
    other = tmp_path / "other-project"  # the same memo copied into a second project
    other.mkdir()
    for directory in (hot_dir, other):
        (directory / "same.md").write_text(NOTE, encoding="utf-8")
    (hot_dir / "own.md").write_text("# 고유한 메모\n\n이 섹션은 한 곳에만 있다.\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", os.pathsep.join(map(str, (hot_dir, other))))
    con = db.connect()
    hot.sync(con)
    con.close()
    log = tmp_path / "recall-log"
    log.mkdir()
    drops = {
        "ambiguous": {"tier": "hot", "path": "/gone/same.md", "heading": "대시보드 메모"},
        "missing": {"tier": "hot", "path": "/gone/never.md", "heading": "없는 섹션"},
        "moved": {"tier": "hot", "path": "/gone/own.md", "heading": "고유한 메모"},
    }
    for name, target in drops.items():
        (log / f"{name}.json").write_text(json.dumps(
            {"query": "고유한 메모 섹션", "target": target, "rank": 1, "limit": 5, "rerank": False},
            ensure_ascii=False), encoding="utf-8")

    ro = dashboard.connect_ro()
    by_file = {d["file"]: d for d in dashboard.recall_log(ro)["drops"]}
    ro.close()
    assert "모호" in by_file["ambiguous.json"]["problem"]
    assert "못 찾음" in by_file["missing.json"]["problem"]
    moved = by_file["moved.json"]
    assert "problem" not in moved and moved["as_recorded"]["rank"] == 1


def test_recall_log_reports_bad_drops_instead_of_failing(store, tmp_path):
    _, db, _, store_module, dashboard = store
    con = db.connect()
    for topic in ("중복 주제 하나", "중복 주제 둘"):
        store_module.write(con, f"{topic}의 본문.", session="t", topic=topic)
    con.close()
    log = tmp_path / "recall-log"
    log.mkdir()
    drops = {
        "null-query": {"query": None, "target": {"tier": "cold", "topic_match": "중복"}},
        "bad-rerank": {"query": "중복", "rerank": "yes", "target": {"tier": "cold", "topic_match": "중복"}},
        "cold-ambiguous": {"query": "중복 주제", "target": {"tier": "cold", "topic_match": "중복 주제"}},
        "cold-missing": {"query": "중복 주제", "target": {"tier": "cold", "topic_match": "없는 주제"}},
    }
    for name, drop in drops.items():
        (log / f"{name}.json").write_text(json.dumps(drop, ensure_ascii=False), encoding="utf-8")
    ro = dashboard.connect_ro()
    by_file = {d["file"]: d["problem"] for d in dashboard.recall_log(ro)["drops"]}
    ro.close()
    assert "query" in by_file["null-query.json"]
    assert "rerank" in by_file["bad-rerank.json"]
    assert "모호" in by_file["cold-ambiguous.json"]
    assert "못 찾음" in by_file["cold-missing.json"]
