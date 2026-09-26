"""What mem_write checks before and after it stores.

A secret is refused outright. Once stored its bytes sit in FTS segments, freed pages
and the WAL, and only `mem_forget` followed by an FTS optimize and a VACUUM gets them
out (see test_forget.py), so a warning after the write would come too late.

A near-duplicate is stored and reported. The usual cause is a memory rewritten a
minute later -- measured on the live store, the two cold pairs at cosine >= 0.95 were
both exactly that -- and the rewrite is often the one worth keeping, so the fix is
`supersedes`, not refusing the write.

The fake credentials are assembled from pieces so no scanner mistakes this file for
a leak.
"""

import importlib

import pytest

AWS_KEY = "AKIA" + "ZQX2" * 4
GITHUB_TOKEN = "ghp_" + "a1B2" * 9
PRIVATE_KEY = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEow..."
PASSWORD = "password" + "=" + "hunter2hunter2"

MEMORY = (
    "리랭크 게이트를 3레인에서 2레인으로 바꿔 보는 실험을 했다. cold R@1 은 그대로였고 "
    "hot R@5 가 한 건 올랐지만 지연이 두 배가 되어 채택하지 않았다."
)
REWRITE = MEMORY + " 측정 조건은 limit=5, rerank auto 였다."
UNRELATED = "워크숍 CFN 템플릿에서 idc 전파 대기를 90초로 늘리고 재시도를 세 번으로 정했다."


@pytest.fixture
def store(tmp_path, monkeypatch, real_model_dir):
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOT", "0")

    from agent_memory import config, db, store as store_module

    for module in (config, db, store_module):
        importlib.reload(module)
    return db, store_module


@pytest.mark.parametrize("secret", [AWS_KEY, GITHUB_TOKEN, PRIVATE_KEY, PASSWORD])
def test_a_secret_is_refused_and_nothing_is_written(store, secret):
    db, store_module = store
    con = db.connect()

    result = store_module.write(con, f"배포 설정을 정리했다. {secret} 는 스테이징 전용이다.")
    assert result["status"] == "error"
    assert secret not in result["message"]  # the refusal must not echo it back
    for table in ("memories", "memories_fts", "memory_chunks"):
        assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    con.close()


def test_a_secret_in_the_topic_is_refused_too(store):
    db, store_module = store
    con = db.connect()
    assert store_module.write(con, MEMORY, topic=f"키 {AWS_KEY}")["status"] == "error"
    con.close()


def test_talking_about_passwords_is_not_a_secret(store):
    db, store_module = store
    con = db.connect()
    result = store_module.write(
        con, "비밀번호 정책을 바꿨다. password 필드는 이제 해시만 담고 token 은 1시간 뒤 만료된다."
    )
    assert result["status"] == "ok"
    con.close()


def test_a_near_duplicate_is_stored_and_reported(store):
    db, store_module = store
    con = db.connect()
    first = store_module.write(con, MEMORY)["memory_id"]
    assert "near_duplicate" not in store_module.write(con, UNRELATED)

    result = store_module.write(con, REWRITE)
    assert result["status"] == "ok"
    assert result["near_duplicate"]["memory_id"] == first
    assert result["near_duplicate"]["similarity"] >= 0.95
    assert "supersedes" in result["warning"]
    assert con.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3
    con.close()


def test_no_duplicate_warning_when_the_rewrite_supersedes_it(store):
    db, store_module = store
    con = db.connect()
    first = store_module.write(con, MEMORY)["memory_id"]
    result = store_module.write(con, REWRITE, supersedes=first)
    assert result["supersedes"]
    assert "near_duplicate" not in result
    assert "warning" not in result
    con.close()
