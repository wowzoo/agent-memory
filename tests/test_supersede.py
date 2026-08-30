"""Marking a memory corrected: surfaced on recall, never applied to ranking."""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(tmp_path / "no-hot"))
    monkeypatch.setenv("AGENT_MEMORY_RERANK_MODEL_DIR", str(tmp_path / "no-reranker"))
    from agent_memory import config, db, hot, rerank, store as store_module

    for module in (config, db, hot, rerank, store_module):
        importlib.reload(module)
    return db, store_module


def two_versions(db, store_module):
    con = db.connect()
    old = store_module.write(con, "빈 장 검사는 별도 스크립트로 신설했다.", session="t")
    new = store_module.write(
        con, "빈 장 검사에 따로 실행할 스크립트는 없다 — 기존 검사에 흡수했다.",
        session="t", supersedes=old["memory_id"][:8],
    )
    return con, old, new


def test_recall_says_which_result_was_corrected(store):
    db, store_module = store
    con, old, new = two_versions(db, store_module)
    results = store_module.recall(con, "빈 장 검사 스크립트", limit=5, rerank=False)["results"]
    by_id = {r["id"]: r for r in results}
    assert by_id[old["memory_id"]]["superseded_by"]["id"] == new["memory_id"]
    # The correction itself carries nothing, and the old row's content is untouched.
    assert "superseded_by" not in by_id[new["memory_id"]]
    assert "별도 스크립트로 신설" in by_id[old["memory_id"]]["content"]
    con.close()


def test_ranking_is_untouched(store):
    """The promise. A penalty would be the rejected hot boost in mirror image, and the
    superseded copy is often the fuller account -- it just is not current."""
    db, store_module = store
    con, old, new = two_versions(db, store_module)
    order = lambda: [  # noqa: E731 — read twice, one expression
        r["id"] for r in store_module.recall(con, "빈 장 검사", limit=5, rerank=False)["results"]
    ]
    before = order()
    # The mark on its own, with nothing written: writing a third memory would move the
    # order by being relevant, which is not what this test is about.
    con.execute("UPDATE memories SET superseded_by = NULL WHERE id = ?", (old["memory_id"],))
    con.commit()
    assert order() == before, "clearing the mark must not move anything either"
    assert store_module.supersede(
        con, old["memory_id"], new["memory_id"]
    )["status"] == "ok"
    assert order() == before
    con.close()


def test_a_hot_section_cannot_be_superseded(store, tmp_path, monkeypatch):
    """Its id is a hash of its body, so the mark would vanish on the next edit of the
    file -- and a file is the source of truth, not an archived copy of one."""
    db, store_module = store
    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    (hot_dir / "notes.md").write_text("## 판정\n\n스크립트를 신설했다.\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(hot_dir))
    from agent_memory import hot as hot_module

    importlib.reload(hot_module)
    con = db.connect()
    hot_module.sync(con)
    section = con.execute("SELECT id FROM memories WHERE tier = 'hot'").fetchone()
    other = store_module.write(con, "그 스크립트는 없앴다.", session="t")
    result = store_module.supersede(con, section["id"], other["memory_id"])
    assert result["status"] == "error"
    assert "edit the file" in result["message"]
    con.close()


def test_a_reference_that_stops_resolving_says_so(store):
    """A dangling reference is information: most often a hot section's heading changed."""
    db, store_module = store
    con, old, new = two_versions(db, store_module)
    con.execute("UPDATE memories SET superseded_by = ? WHERE id = ?",
                ("~/gone.md#없는 절", old["memory_id"]))
    con.commit()
    results = store_module.recall(con, "빈 장 검사 스크립트", limit=5, rerank=False)["results"]
    block = next(r["superseded_by"] for r in results if r["id"] == old["memory_id"])
    assert block["resolved"] is False and block["ref"] == "~/gone.md#없는 절"
    con.close()


def test_write_reports_a_reference_it_could_not_use(store):
    """The memory is stored either way: losing a write because a pointer missed would
    be the worse failure, so the miss is reported next to the id."""
    db, store_module = store
    con = db.connect()
    result = store_module.write(con, "정정본", session="t", supersedes="zzzzzzzz")
    assert result["status"] == "ok" and result["memory_id"]
    assert "no single memory matches" in result["supersedes"]
    con.close()


def test_a_memory_cannot_supersede_itself(store):
    db, store_module = store
    con = db.connect()
    one = store_module.write(con, "자기 참조 검사", session="t")
    assert store_module.supersede(
        con, one["memory_id"], one["memory_id"]
    )["status"] == "error"
    con.close()


def test_a_hot_section_can_be_the_correction(store, tmp_path, monkeypatch):
    """`path#heading` rather than an id, because a hot id dies on the next body edit --
    which is exactly the case this feature exists for: the file corrected the archive."""
    db, store_module = store
    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    note = hot_dir / "notes.md"
    note.write_text("## 판정 — 스크립트는 없다\n\n기존 검사에 흡수했다.\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(hot_dir))
    from agent_memory import hot as hot_module

    importlib.reload(hot_module)
    con = db.connect()
    hot_module.sync(con)
    old = store_module.write(con, "빈 장 검사는 별도 스크립트로 신설했다.", session="t")
    ref = f"{note}#판정 — 스크립트는 없다"
    assert store_module.supersede(con, old["memory_id"], ref)["status"] == "ok"

    # Edit the body: the id changes, the reference still resolves.
    note.write_text("## 판정 — 스크립트는 없다\n\n네 번째 검사로 흡수했다.\n", encoding="utf-8")
    hot_module.sync(con)
    results = store_module.recall(con, "빈 장 검사 스크립트", limit=5, rerank=False)["results"]
    block = next(r["superseded_by"] for r in results if r["id"] == old["memory_id"])
    assert block["tier"] == "hot" and block["path"] == str(note)
    con.close()
