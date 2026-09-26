"""Which agent wrote a memory, in a store shared by claude, codex and kiro.

Three things have to hold once more than one harness writes here, and each of them was
a way to lose information silently:

  - a cold row records its writer, and `agent=` finds it again
  - a hot row records whose *directory* the file is in, so a Codex recall that mirrors
    Claude's memory files does not relabel them (the same trap `project` fell into,
    schema v6)
  - the active session is per agent, so one harness naming a session does not decide
    where another harness's next unnamed write lands
"""

import importlib
from pathlib import Path

import pytest

MEMORY_A = "# 첫 번째 주제\n\n한국어 검색이 안 되는 문제를 고쳤다. 임베딩 모델이 원인이었다.\n"


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("AGENT_MEMORY_AGENT", raising=False)
    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(hot_dir))

    from agent_memory import config, db, hot, store as store_module

    for module in (config, db, hot, store_module):
        importlib.reload(module)
    # The client name is process state, so it survives a reload of nothing else --
    # clear it explicitly or the previous test's harness leaks into this one.
    store_module.set_client(None)
    store_module._client = None
    return hot_dir, db, hot, store_module


def test_env_overrides_the_client_name(store, monkeypatch):
    _, _, _, store_module = store
    store_module.set_client("Claude Code")
    assert store_module.detect_agent() == "claude-code"

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "codex")
    assert store_module.detect_agent() == "codex"


def test_client_name_is_normalized(store):
    _, _, _, store_module = store
    assert store_module.normalize_agent("  Kiro CLI  ") == "kiro-cli"
    assert store_module.normalize_agent("") is None
    assert store_module.normalize_agent(None) is None


def test_write_records_the_agent_and_recall_filters_by_it(store, monkeypatch):
    _, db, _, store_module = store
    con = db.connect()

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude")
    written = store_module.write(con, "리랭커 게이트를 채택했다.")
    assert written["agent"] == "claude"

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "codex")
    store_module.write(con, "리랭커 게이트를 확인했다.")

    found = store_module.recall(con, "리랭커 게이트", agent="claude", tier="cold")
    assert [r["id"] for r in found["results"]] == [written["memory_id"]]
    assert found["results"][0]["agent"] == "claude"

    # Case and spacing are normalized on the way in, so they are on the way out too.
    assert store_module.recall(con, "리랭커 게이트", agent="CLAUDE", tier="cold")[
        "results_count"
    ] == 1
    both = store_module.recall(con, "리랭커 게이트", tier="cold")
    assert {r["agent"] for r in both["results"]} == {"claude", "codex"}


def test_unattributed_writes_stay_null(store):
    _, db, _, store_module = store
    con = db.connect()
    written = store_module.write(con, "출처를 모르는 기억이다.")
    assert written["agent"] is None
    row = con.execute(
        "SELECT agent FROM memories WHERE id = ?", (written["memory_id"],)
    ).fetchone()
    assert row["agent"] is None


def test_active_session_is_per_agent(store, monkeypatch):
    _, db, _, store_module = store
    con = db.connect()

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude")
    store_module.write(con, "첫 기억.", session="claude-chronicle")

    # Codex names its own session; Claude's next unnamed write must not follow it.
    monkeypatch.setenv("AGENT_MEMORY_AGENT", "codex")
    store_module.write(con, "코덱스 기억.", session="codex-chronicle")
    assert store_module.write(con, "코덱스 두 번째.")["session"] == "codex-chronicle"

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude")
    assert store_module.write(con, "클로드 두 번째.")["session"] == "claude-chronicle"
    assert store_module.active_session(con, "claude") == "claude-chronicle"


def test_legacy_active_session_is_the_fallback(store, monkeypatch):
    """A store written before per-agent keys keeps its session for the first write."""
    _, db, _, store_module = store
    con = db.connect()
    db.set_meta(con, "active_session", "이관-전-세션")

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude")
    assert store_module.write(con, "첫 기억.")["session"] == "이관-전-세션"
    # Naming one moves this agent off the legacy key without touching it.
    store_module.write(con, "둘째.", session="claude-chronicle")
    assert db.get_meta(con, "active_session") == "이관-전-세션"
    monkeypatch.setenv("AGENT_MEMORY_AGENT", "codex")
    assert store_module.write(con, "코덱스 첫 기억.")["session"] == "이관-전-세션"


def test_hot_sections_inherit_the_directory_owner(store, monkeypatch):
    hot_dir, db, hot, store_module = store
    (hot_dir / "a.md").write_text(MEMORY_A, encoding="utf-8")

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude")
    con = db.connect()
    assert hot.sync(con)["indexed"] == 1
    assert {r["agent"] for r in con.execute("SELECT agent FROM hot_files")} == {"claude"}

    # A different harness mirrors the same directory. It must not claim it: whose memory
    # files these are is a fact about the directory, not about who ran the recall.
    monkeypatch.setenv("AGENT_MEMORY_AGENT", "codex")
    (hot_dir / "a.md").write_text(MEMORY_A + "\n# 새 섹션\n\n내용이 늘었다.\n", encoding="utf-8")
    assert hot.sync(con)["indexed"] == 1
    agents = {
        r["agent"] for r in con.execute("SELECT agent FROM memories WHERE tier = 'hot'")
    }
    assert agents == {"claude"}
    assert store_module.recall(con, "한국어 검색", agent="codex", tier="hot")[
        "results_count"
    ] == 0


def test_a_claude_directory_first_indexed_by_codex_is_still_claude(store, monkeypatch, tmp_path):
    """Inheriting the owner only works once someone has named it. A directory under
    ~/.claude/projects that a Codex recall indexes first -- which is what happened to
    four directories after the 2026-09-07 move -- is Claude Code's by where it lives,
    and every file added to it later inherited 'codex' from that first index."""
    _, db, hot, _ = store
    monkeypatch.setenv("HOME", str(tmp_path))
    claude_dir = tmp_path / ".claude" / "projects" / "-p" / "memory"
    claude_dir.mkdir(parents=True)
    (claude_dir / "a.md").write_text(MEMORY_A, encoding="utf-8")
    monkeypatch.setenv("AGENT_MEMORY_HOT_PATHS", str(claude_dir))

    monkeypatch.setenv("AGENT_MEMORY_AGENT", "codex")
    con = db.connect()
    assert hot.sync(con)["indexed"] == 1
    assert {r["agent"] for r in con.execute("SELECT agent FROM hot_files")} == {"claude"}
    assert {
        r["agent"] for r in con.execute("SELECT agent FROM memories WHERE tier = 'hot'")
    } == {"claude"}


def test_migration_adds_the_columns_to_an_existing_store(store):
    """A v7 store gains both columns, and its hot rows are attributed from their path."""
    _, db, _, _ = store
    con = db.connect()
    con.execute("ALTER TABLE memories DROP COLUMN agent")
    con.execute("ALTER TABLE hot_files DROP COLUMN agent")
    claude_dir = Path.home() / ".claude" / "projects" / "-p" / "memory"
    con.execute(
        "INSERT INTO memories(id, session, topic, content, created_at, source, tier, "
        "path) VALUES ('h', 's', 't', 'c', '2026-01-01', 'hot', 'hot', ?)",
        (f"{claude_dir}/x.md",),
    )
    con.execute(
        "INSERT INTO hot_files(path, mtime, size, hash) VALUES (?, 0, 0, 'd')",
        (f"{claude_dir}/x.md",),
    )
    con.commit()
    con.close()

    con = db.connect()
    assert db.get_meta(con, "schema_version") == str(db.SCHEMA_VERSION)
    assert con.execute("SELECT agent FROM hot_files").fetchone()["agent"] == "claude"
    assert con.execute("SELECT agent FROM memories").fetchone()["agent"] == "claude"
