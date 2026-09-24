"""Bounds on what a caller can ask for.

Neither was bounded before 2026-09-24. `limit` only sizes the response -- the fused
candidate list is already capped by the lane pools -- but at `detail=3` that is every
candidate's full text in one tool result. `content` had no ceiling at all, and every
character of it is tokenized and embedded chunk by chunk inside the write.
"""

import asyncio
import importlib

import pytest
from mcp.server.mcpserver.exceptions import ToolError


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOT", "0")

    from agent_memory import config, db, store as store_module

    for module in (config, db, store_module):
        importlib.reload(module)
    return db, store_module


@pytest.mark.parametrize("limit", [0, -1, 51])
def test_recall_limit_is_bounded_at_the_tool(store, limit):
    from agent_memory import server

    with pytest.raises(ToolError, match="limit"):
        asyncio.run(server.mcp.call_tool("mem_recall", {"query": "검색", "limit": limit}))


def test_oversized_content_is_refused_and_nothing_is_written(store, monkeypatch):
    db, store_module = store
    monkeypatch.setattr(store_module.config, "MAX_CONTENT_CHARS", 100)
    con = db.connect()

    result = store_module.write(con, "가" * 101)
    assert result["status"] == "error"
    assert "100" in result["message"]
    assert con.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0

    assert store_module.write(con, "가" * 100)["status"] == "ok"
    con.close()


def test_the_cap_counts_what_is_stored(store, monkeypatch):
    """Surrounding whitespace is stripped before storing, so it is not held against
    the caller."""
    db, store_module = store
    monkeypatch.setattr(store_module.config, "MAX_CONTENT_CHARS", 100)
    con = db.connect()
    assert store_module.write(con, "\n\n" + "가" * 100 + "\n\n")["status"] == "ok"
    con.close()
