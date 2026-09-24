"""Forgetting a cold memory: gone from every table, and gone from the file.

The second half is the reason this exists. The common reason to delete a memory is
that it should never have been written -- a key, a password, someone's details -- and
a plain DELETE does not remove those bytes. Measured 2026-09-24 with a planted secret:
DELETE + checkpoint left 3 copies; secure_delete + FTS optimize still left 1 on this
very fixture, on a freed page. FTS optimize + VACUUM leaves none.
"""

import importlib

import pytest

SECRET = "AKIAZQXSECRET9912"
# The FTS index stores terms lowercased and prefix-compressed, so the full string is
# not a reliable probe. This fragment survives both.
PROBES = (b"ZQXSECRET", b"zqxsecret")


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
    return tmp_path, hot_dir, db, hot, store_module


def count(con, sql, *params):
    return con.execute(sql, params).fetchone()[0]


def test_forget_removes_the_memory_from_every_table(store):
    _, _, db, _, store_module = store
    con = db.connect()
    keep = store_module.write(
        con, "형태소 분석기는 Kiwi 로 정했다.", entities=["Kiwi", "형태소 분석"],
        relations=[{"source": "Kiwi", "predicate": "implements", "target": "형태소 분석"}],
    )["memory_id"]
    gone = store_module.write(
        con, "배포 키 기록은 잘못 남긴 것이다.", entities=["배포 키", "Kiwi"],
        relations=[{"source": "배포 키", "predicate": "uses", "target": "Kiwi"}],
    )["memory_id"]

    result = store_module.forget(con, gone[:8])
    assert result["status"] == "ok"
    assert result["memory_id"] == gone
    assert result["removed"]["relations"] == 1
    assert result["removed"]["chunks"] >= 1

    for table, column in (("memories", "id"), ("memories_fts", "memory_id"),
                          ("memory_chunks", "memory_id"), ("memory_entities", "memory_id"),
                          ("relations", "memory_id")):
        assert count(con, f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", gone) == 0
    # An entity only the forgotten memory used goes; a shared one stays.
    names = {r[0] for r in con.execute("SELECT name FROM entities")}
    assert "배포 키" not in names and "kiwi" in {n.lower() for n in names}
    assert count(con, "SELECT COUNT(*) FROM relations WHERE memory_id = ?", keep) == 1

    hits = store_module.recall(con, "배포 키 기록", limit=5, rerank=False)["results"]
    assert gone not in [h["id"] for h in hits]
    con.close()


def test_the_bytes_are_gone_from_the_file(store):
    tmp_path, _, db, _, store_module = store
    con = db.connect()
    for i in range(40):
        store_module.write(con, f"평범한 기록 {i}. 한국어 검색을 실측했다. akia{i} " * 5)
    gone = store_module.write(con, f"배포 키를 적어 둔다: {SECRET} 는 스테이징 전용이다.")
    for i in range(40, 60):
        store_module.write(con, f"뒤에 쌓인 기록 {i}. " * 5)

    result = store_module.forget(con, gone["memory_id"])
    assert result["scrubbed"] is True
    con.close()

    files = sorted(tmp_path.glob("memory.db*"))
    assert files
    leaks = {f.name: sum(f.read_bytes().count(p) for p in PROBES) for f in files}
    assert set(leaks.values()) == {0}, leaks


@pytest.mark.parametrize("ref", ["", "abc", "0000000000000000"])
def test_a_short_or_unmatched_ref_deletes_nothing(store, ref):
    _, _, db, _, store_module = store
    con = db.connect()
    store_module.write(con, "지워지면 안 되는 기록이다.")
    assert store_module.forget(con, ref)["status"] == "error"
    assert count(con, "SELECT COUNT(*) FROM memories") == 1
    con.close()


def test_a_hot_section_is_refused(store):
    _, hot_dir, db, hot, store_module = store
    (hot_dir / "note.md").write_text("# 결정\n\nhot 은 파일이 원본이다.\n", encoding="utf-8")
    con = db.connect()
    hot.sync(con)
    hot_id = con.execute("SELECT id FROM memories WHERE tier = 'hot'").fetchone()[0]

    result = store_module.forget(con, hot_id)
    assert result["status"] == "error"
    assert "note.md" in result["message"]
    assert count(con, "SELECT COUNT(*) FROM memories WHERE id = ?", hot_id) == 1
    con.close()


def test_a_correction_that_is_forgotten_is_reported_not_rewritten(store):
    """The older memory is still out of date after its correction is forgotten -- often
    the correction was forgotten *because* it leaked something. So its mark is left,
    it renders as unresolved, and the caller is told which memories point at a gap."""
    _, _, db, _, store_module = store
    con = db.connect()
    old = store_module.write(con, "빈 장 검사는 별도 스크립트다.")["memory_id"]
    new = store_module.write(con, "빈 장 검사는 기존 검사에 흡수했다.",
                             supersedes=old[:8])["memory_id"]
    # `supersedes=` marks the *old* row with the new memory's full id.
    assert store_module.forget(con, new)["superseded_refs"] == [old]

    hits = store_module.recall(con, "빈 장 검사", limit=5, rerank=False)["results"]
    mark = {h["id"]: h for h in hits}[old]["superseded_by"]
    assert mark["resolved"] is False
    con.close()


def _payload(result) -> dict:
    """The body of an in-process `call_tool` result (a CallToolResult)."""
    import json

    if result.structured_content is not None:
        body = result.structured_content
        return body.get("result", body) if set(body) == {"result"} else body
    return json.loads(result.content[0].text)


def test_the_tool_deletes_and_scrubs_through_the_server(store):
    """Through the MCP layer, not just `store.forget`: the server opens its connection
    with `with db.connect()`, and VACUUM refuses to run inside a transaction."""
    import asyncio

    from agent_memory import server

    tmp_path, _, db, _, _ = store

    async def run():
        for i in range(30):
            await server.mcp.call_tool("mem_write", {"content": f"기록 {i}. 한국어 검색 실측. " * 5})
        written = _payload(await server.mcp.call_tool(
            "mem_write", {"content": f"키: {SECRET} 스테이징 전용"}))
        forgotten = _payload(await server.mcp.call_tool(
            "mem_forget", {"memory_id": written["memory_id"][:8]}))
        refused = _payload(await server.mcp.call_tool("mem_forget", {"memory_id": "abc"}))
        return written, forgotten, refused

    written, forgotten, refused = asyncio.run(run())
    assert forgotten["status"] == "ok" and forgotten["scrubbed"] is True
    assert forgotten["memory_id"] == written["memory_id"]
    assert refused["status"] == "error"

    con = db.connect()
    assert con.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 30
    con.close()
    leaks = {f.name: sum(f.read_bytes().count(p) for p in PROBES)
             for f in tmp_path.glob("memory.db*")}
    assert set(leaks.values()) == {0}, leaks
