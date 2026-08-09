"""Re-indexing a changed hot file must leave no orphan index rows.

The regression this guards: memories_fts is a standalone FTS5 table, so deleting a
memory row without deleting its index entry leaves an orphan. SQLite then reuses that
rowid for an unrelated memory and the lexical lane returns the wrong record — recall
quality collapsed to R@5 0% before an IntegrityError finally surfaced the cause. The
fix was to join on an explicit memory_id instead of relying on rowid parity, so these
tests assert the invariant directly rather than trusting the crash to reappear.
"""

import os

import pytest

MEMORY_A = "# 첫 번째 주제\n\n한국어 검색이 안 되는 문제를 고쳤다. 임베딩 모델이 원인이었다.\n"
MEMORY_B = "# 두 번째 주제\n\nAI-DLC v2 패치를 4트리에 재적용하고 doctor 결과를 확인했다.\n"
REPLACED = "# 완전히 다른 주제\n\n대시보드를 처음부터 새로 구현하고 커밋했다.\n"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(hot_dir))

    # config reads the environment at import time, so reload it and its dependents.
    import importlib

    from agent_memory import config, db, hot, store as store_module

    for module in (config, db, hot, store_module):
        importlib.reload(module)
    return hot_dir, db, hot, store_module


def orphan_count(con) -> int:
    return con.execute(
        "SELECT COUNT(*) FROM memories_fts f "
        "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = f.memory_id)"
    ).fetchone()[0]


def test_reindex_leaves_no_orphans(store):
    hot_dir, db, hot, _ = store
    (hot_dir / "a.md").write_text(MEMORY_A, encoding="utf-8")
    (hot_dir / "b.md").write_text(MEMORY_B, encoding="utf-8")

    con = db.connect()
    assert hot.sync(con)["indexed"] == 2
    assert orphan_count(con) == 0

    # Rewriting a file drops its old sections and indexes the new ones.
    os.utime(hot_dir / "a.md", (0, 0))
    (hot_dir / "a.md").write_text(REPLACED, encoding="utf-8")
    assert hot.sync(con)["indexed"] == 1
    assert orphan_count(con) == 0

    # Deleting a file removes the mirror entirely.
    (hot_dir / "b.md").unlink()
    assert hot.sync(con)["removed"] == 1
    assert orphan_count(con) == 0

    fts_rows = con.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
    memories = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert fts_rows == memories
    con.close()


def test_lexical_returns_the_right_record_after_reindex(store):
    """The orphan bug's real symptom: a stale index row aliasing a live memory."""
    hot_dir, db, hot, store_module = store
    (hot_dir / "a.md").write_text(MEMORY_A, encoding="utf-8")
    (hot_dir / "b.md").write_text(MEMORY_B, encoding="utf-8")
    con = db.connect()
    hot.sync(con)

    os.utime(hot_dir / "a.md", (0, 0))
    (hot_dir / "a.md").write_text(REPLACED, encoding="utf-8")
    hot.sync(con)

    # "임베딩" existed only in the replaced text, so nothing may match it now.
    assert store_module.recall(con, "임베딩 모델", mode="lexical")["results"] == []

    # A term from the replacement must resolve to the file that now contains it.
    hits = store_module.recall(con, "대시보드", mode="lexical")["results"]
    assert hits, "replacement text should be searchable"
    assert all(h["path"].endswith("a.md") for h in hits)
    assert "대시보드" in hits[0]["content"]
    con.close()


def test_unchanged_files_are_not_reindexed(store):
    hot_dir, db, hot, _ = store
    (hot_dir / "a.md").write_text(MEMORY_A, encoding="utf-8")
    con = db.connect()
    hot.sync(con)
    again = hot.sync(con)
    assert again == {"indexed": 0, "sections": 0, "removed": 0, "unchanged": 1}
    con.close()
