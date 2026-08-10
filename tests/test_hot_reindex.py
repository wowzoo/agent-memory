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
def store(tmp_path, monkeypatch, real_model_dir):
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
    assert again == {
        "indexed": 0, "sections": 0, "kept": 0, "removed": 0, "unchanged": 1
    }
    con.close()


TWO_SECTIONS = (
    "# 첫 번째 주제\n\n한국어 검색이 안 되는 문제를 고쳤다.\n\n"
    "# 두 번째 주제\n\nAI-DLC v2 패치를 4트리에 재적용했다.\n"
)
EDITED_SECOND = (
    "# 첫 번째 주제\n\n한국어 검색이 안 되는 문제를 고쳤다.\n\n"
    "# 두 번째 주제\n\n패치를 4트리에 재적용하고 doctor 결과까지 확인했다.\n"
)
INSERTED_ABOVE = (
    "# 새로 끼운 주제\n\n평가셋을 20개에서 72개로 늘렸다.\n\n" + TWO_SECTIONS
)


def annotated(con, memory_id: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM memory_entities WHERE memory_id = ?", (memory_id,)
        ).fetchone()
    )


def section_id(con, heading: str) -> str:
    return con.execute(
        "SELECT id FROM memories WHERE tier = 'hot' AND topic = ?", (heading,)
    ).fetchone()["id"]


def test_editing_one_section_keeps_the_others_annotations(store):
    """Graph annotations cascade off memory_id, so re-inserting an untouched section
    would silently delete work that took a full read of the memory to produce."""
    hot_dir, db, hot, _ = store
    (hot_dir / "a.md").write_text(TWO_SECTIONS, encoding="utf-8")
    con = db.connect()
    hot.sync(con)

    from agent_memory import graph

    untouched = section_id(con, "첫 번째 주제")
    edited = section_id(con, "두 번째 주제")
    for memory_id in (untouched, edited):
        graph.attach(con, memory_id, [{"name": f"엔티티-{memory_id[:4]}"}], [])
    assert annotated(con, untouched) and annotated(con, edited)

    os.utime(hot_dir / "a.md", (0, 0))
    (hot_dir / "a.md").write_text(EDITED_SECOND, encoding="utf-8")
    stats = hot.sync(con)
    assert stats == {
        "indexed": 1, "sections": 2, "kept": 1, "removed": 0, "unchanged": 0
    }

    # The untouched section kept its id, its vectors and its annotations; the edited
    # one is a different memory now, so its annotations are gone rather than reused.
    assert section_id(con, "첫 번째 주제") == untouched
    assert annotated(con, untouched)
    assert section_id(con, "두 번째 주제") != edited
    assert not annotated(con, edited)
    assert orphan_count(con) == 0
    con.close()


FRONTMATTER = (
    "---\nname: a\nmetadata:\n  modified: 2026-08-09T10:49:40.780Z\n---\n\n"
    "# 첫 번째 주제\n\n한국어 검색이 안 되는 문제를 고쳤다.\n"
)
FRONTMATTER_TOUCHED = FRONTMATTER.replace("10:49:40.780Z", "11:02:11.912Z")


def test_frontmatter_is_not_part_of_the_section(store):
    """`modified:` is rewritten whenever the file is touched. With it in the body, the
    section hashed differently every time and took its annotations down with it."""
    hot_dir, db, hot, _ = store
    (hot_dir / "a.md").write_text(FRONTMATTER, encoding="utf-8")
    con = db.connect()
    hot.sync(con)

    from agent_memory import graph

    memory_id = section_id(con, "첫 번째 주제")
    graph.attach(con, memory_id, ["엔티티"], [])
    body = con.execute(
        "SELECT content FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()["content"]
    assert "modified" not in body and "name: a" not in body

    os.utime(hot_dir / "a.md", (0, 0))
    (hot_dir / "a.md").write_text(FRONTMATTER_TOUCHED, encoding="utf-8")
    assert hot.sync(con)["kept"] == 1
    assert section_id(con, "첫 번째 주제") == memory_id
    assert annotated(con, memory_id)
    con.close()


def test_another_projects_hot_files_survive_a_sync(store, monkeypatch, tmp_path):
    """One store, several registered projects: each server process only sees its own
    project's memory directory, so every other project's rows look like files that
    disappeared. Dropping them takes their graph annotations with them and no
    re-index brings those back -- which made alternating recalls between two projects
    destroy each other's annotation work on every call."""
    hot_dir, db, hot, _ = store
    other = tmp_path / "other-project-hot"
    other.mkdir()
    (hot_dir / "mine.md").write_text(MEMORY_A, encoding="utf-8")
    (other / "theirs.md").write_text(MEMORY_B, encoding="utf-8")

    from agent_memory import graph

    con = db.connect()
    # The other project's server indexes and annotates its own file.
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(other))
    assert hot.sync(con)["indexed"] == 1
    theirs = section_id(con, "두 번째 주제")
    graph.attach(con, theirs, [{"name": "남의 엔티티"}], [])
    assert annotated(con, theirs)

    # Now this project's server runs. It must index its own file and leave theirs be.
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(hot_dir))
    stats = hot.sync(con)
    assert stats["indexed"] == 1
    assert stats["removed"] == 0, "another project's files are not ours to drop"
    assert section_id(con, "두 번째 주제") == theirs
    assert annotated(con, theirs)
    assert section_id(con, "첫 번째 주제")

    # A file that vanishes from *our* directory is still dropped.
    (hot_dir / "mine.md").unlink()
    assert hot.sync(con)["removed"] == 1
    assert annotated(con, theirs)
    con.close()


def test_inserting_a_section_does_not_renumber_the_rest(store):
    """Position is not part of the id: adding a section above must not invalidate the
    sections below it, which is what an index-keyed id did."""
    hot_dir, db, hot, _ = store
    (hot_dir / "a.md").write_text(TWO_SECTIONS, encoding="utf-8")
    con = db.connect()
    hot.sync(con)
    before = {section_id(con, h) for h in ("첫 번째 주제", "두 번째 주제")}

    os.utime(hot_dir / "a.md", (0, 0))
    (hot_dir / "a.md").write_text(INSERTED_ABOVE, encoding="utf-8")
    stats = hot.sync(con)
    assert stats["sections"] == 3 and stats["kept"] == 2
    assert before == {section_id(con, h) for h in ("첫 번째 주제", "두 번째 주제")}
    assert orphan_count(con) == 0
    con.close()
