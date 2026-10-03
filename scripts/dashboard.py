"""A read-only dashboard over the store: health checks and a memory browser.

    uv run python scripts/dashboard.py              # http://127.0.0.1:8765
    uv run python scripts/dashboard.py --port 9000

Read-only is the whole contract, and two obvious shortcuts break it:

- `db.connect()` is not a reader. It creates the directory, runs the schema script,
  migrates and rewrites `meta` before it hands the connection back. This opens the file
  with `mode=ro` instead, which also refuses to create a store that is not there.
- `store.recall()` calls `hot.sync()` whenever `tier != "cold"`, which re-indexes memory
  files from disk. `main()` turns `config.HOT_ENABLED` off for the server process, so the
  recall tab searches the mirror as it stands; the mirror catches up on the next real
  recall. Called from elsewhere with sync on, `recall()` here fails on the read-only
  connection instead of writing.

A connection lives for one request. A long-lived read transaction on a WAL database holds
back checkpointing for every writer sharing the file.

Missing hot sources are classified by `prune_moved_hot.plan()` itself, not a copy of its
rules, so this page and the tool that acts on it cannot disagree. A row with no live twin
is the last copy of what its file said -- eleven were kept on purpose on 2026-09-13
(`docs/measurements.md`, `### 정리 후 기준선`) -- so it is reported, never offered for deletion.
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_memory import config, db, hot, rerank, store  # noqa: E402
import collect_queries  # noqa: E402
from prune_moved_hot import plan  # noqa: E402

PAGE = Path(__file__).with_name("dashboard.html")
PAGE_SIZE = 50



def connect_ro() -> sqlite3.Connection:
    con = sqlite3.connect(f"{config.DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _count(con, sql: str, params=()) -> int:
    return con.execute(sql, params).fetchone()[0]


def _check(label, level, value, note=None, rows=None) -> dict:
    """level: 'error' = an invariant is broken, 'warn' = needs a look, 'ok', 'info'."""
    return {"label": label, "level": level, "value": value, "note": note, "rows": rows or []}


def status(con) -> dict:
    total = _count(con, "SELECT COUNT(*) FROM memories")
    checks = []

    mismatch = db.embedding_mismatch(con)
    model_ok = (config.MODEL_DIR / "model.onnx").exists() and (
        config.MODEL_DIR / "tokenizer.json").exists()
    checks.append(_check(
        "임베딩 모델", "error" if mismatch or not model_ok else "ok",
        db.get_meta(con, "embed_model"),
        mismatch or (None if model_ok else f"{config.MODEL_DIR} 에 export 가 없다")))
    checks.append(_check(
        "리랭커", "ok" if rerank.available() else "warn",
        "있음" if rerank.available() else "없음",
        None if rerank.available() else f"{config.RERANK_MODEL_DIR} — 회수가 리랭크 없이 돈다"))

    unembedded = _count(con, "SELECT COUNT(*) FROM memories m WHERE NOT EXISTS "
                             "(SELECT 1 FROM memory_chunks c WHERE c.memory_id = m.id)")
    checks.append(_check("임베딩 커버리지", "error" if unembedded else "ok",
                         f"{total - unembedded:,} / {total:,}"))

    # memory_id is UNINDEXED in FTS5, so a NOT EXISTS against it scans per row.
    ids = {r[0] for r in con.execute("SELECT id FROM memories")}
    fts = Counter(r[0] for r in con.execute("SELECT memory_id FROM memories_fts"))
    fts_missing = len(ids - fts.keys())
    fts_orphan = len(fts.keys() - ids)
    fts_dup = sum(1 for n in fts.values() if n > 1)
    checks.append(_check(
        "FTS 정합성", "error" if fts_missing or fts_orphan or fts_dup else "ok",
        f"누락 {fts_missing} · 고아 {fts_orphan} · 중복 {fts_dup}"))

    orphans = {
        "청크": "SELECT COUNT(*) FROM memory_chunks c WHERE NOT EXISTS "
               "(SELECT 1 FROM memories m WHERE m.id = c.memory_id)",
        "엔티티 링크": "SELECT COUNT(*) FROM memory_entities me WHERE NOT EXISTS "
                   "(SELECT 1 FROM memories m WHERE m.id = me.memory_id) OR NOT EXISTS "
                   "(SELECT 1 FROM entities e WHERE e.name = me.entity)",
        "관계": "SELECT COUNT(*) FROM relations r WHERE NOT EXISTS "
              "(SELECT 1 FROM memories m WHERE m.id = r.memory_id) OR NOT EXISTS "
              "(SELECT 1 FROM entities e WHERE e.name = r.source) OR NOT EXISTS "
              "(SELECT 1 FROM entities e WHERE e.name = r.target)",
    }
    counts = {name: _count(con, sql) for name, sql in orphans.items()}
    checks.append(_check("고아 행", "error" if any(counts.values()) else "ok",
                         " · ".join(f"{k} {v}" for k, v in counts.items())))

    unresolved, chained = [], 0
    marked = con.execute("SELECT * FROM memories WHERE superseded_by IS NOT NULL").fetchall()
    for row in marked:
        block = store._supersession(con, row) or {}
        if block.get("resolved") is False:
            unresolved.append({"id": row["id"], "topic": row["topic"], "detail": block["ref"]})
        elif "hops" in block:
            chained += 1
    checks.append(_check(
        "정정 참조", "error" if unresolved else "ok",
        f"{len(marked)}건 중 미해석 {len(unresolved)}",
        f"2단계 이상 체인 {chained}건 — 결과에 현재판이 함께 실린다", unresolved))

    changed, missing = [], []
    for row in con.execute("SELECT path, hash FROM hot_files ORDER BY path"):
        path = Path(row["path"])
        if not path.exists():
            missing.append(row["path"])
        elif hot._digest(path)[2] != row["hash"]:
            changed.append({"id": "", "topic": path.name, "detail": row["path"]})
    checks.append(_check(
        "hot 원본 변경", "warn" if changed else "ok", f"{len(changed)}개 파일",
        "색인 뒤 디스크에서 바뀌었다. 다음 recall 의 sync 가 반영한다 — 이 페이지는 sync 하지 않는다.",
        changed))

    move, drop, keep = plan(con)
    reason = {"duplicate": "같은 본문 사본이 여럿 살아 있음",
              "stale body": "대응 섹션은 있으나 본문이 다름"}
    gone = ([(i, "같은 본문 사본이 살아 있음") for i, _ in move]
            + [(i, reason.get(r, r)) for i, r in drop]
            + [(i, "대응 섹션 없음 — 스토어에만 남은 마지막 사본") for i, _ in keep])
    topics = {r["id"]: r for r in con.execute(
        "SELECT id, topic, path FROM memories WHERE tier = 'hot'")}
    checks.append(_check(
        "hot 원본 경로 없음", "warn" if gone else "ok",
        f"{len(missing)}개 파일 · {len(gone)}행",
        "분류는 scripts/prune_moved_hot.plan() 그대로. 마지막 사본은 지우지 말 것 "
        "(09-13 에 일부러 남긴 행이 있다).",
        [{"id": i, "topic": topics[i]["topic"], "detail": f"{why} · {topics[i]['path']}"}
         for i, why in gone]))

    by_tier = {r["tier"]: (r["n"], r["annotated"]) for r in con.execute(
        "SELECT tier, COUNT(*) n, SUM(EXISTS (SELECT 1 FROM memory_entities me "
        "WHERE me.memory_id = m.id)) annotated FROM memories m GROUP BY tier")}
    types = Counter(r[0] for r in con.execute("SELECT type FROM entities"))
    stats = [
        _check("엔티티 주석 커버리지", "info",
               " · ".join(f"{t} {a:,}/{n:,} ({a / n:.0%})" for t, (n, a) in by_tier.items()),
               "hot 도 주석을 받을 수 있다. 섹션 본문을 고치면 id 가 바뀌어 주석이 떨어진다."),
        _check("엔티티 type", "info", f"{len(types)}종 · 한 번만 쓰인 type {sum(1 for n in types.values() if n == 1)}종",
               "type 은 자유 형식이다.",
               [{"id": "", "topic": t, "detail": f"{n}"} for t, n in types.most_common()]),
        _check("관계 없는 엔티티", "info", _count(
            con, "SELECT COUNT(*) FROM entities e WHERE NOT EXISTS (SELECT 1 FROM relations r "
                 "WHERE r.source = e.name) AND NOT EXISTS (SELECT 1 FROM relations r "
                 "WHERE r.target = e.name)")),
        _check("agent 귀속 없음", "info", _count(con, "SELECT COUNT(*) FROM memories WHERE agent IS NULL"),
               "컬럼 도입 전 cold 기록 — 지어내지 않고 비워 둔 것"),
        _check("정정된 기억", "info", len(marked)),
    ]

    wal = config.DB_PATH.with_name(config.DB_PATH.name + "-wal")
    size = config.DB_PATH.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
    return {
        "summary": {
            "memories": total,
            "chunks": _count(con, "SELECT COUNT(*) FROM memory_chunks"),
            "entities": _count(con, "SELECT COUNT(*) FROM entities"),
            "relations": _count(con, "SELECT COUNT(*) FROM relations"),
            "model": f"{db.get_meta(con, 'embed_model')}/{db.get_meta(con, 'embed_dim')}",
            "schema": db.get_meta(con, "schema_version"),
            "size_mb": round(size / 1e6, 1),
            "db": str(config.DB_PATH),
        },
        "checks": checks,
        "stats": stats,
        "performance": performance_note(),
    }


def performance_note() -> str:
    """README's `## 지금 성능` paragraphs, verbatim.

    Quoted rather than re-typed: a number copied here would be one more place to go
    stale, and the README is where the conditions sit next to it.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.find("## 지금 성능")
    if start < 0:
        return ""
    end = text.find("**조건·근거", start)
    return text[start + len("## 지금 성능"):end if end > 0 else None].strip()


def overview(con) -> dict:
    def rows(sql):
        return [dict(r) for r in con.execute(sql)]

    return {
        "by_project": rows(
            "SELECT COALESCE(project, '(없음)') project, "
            "SUM(tier = 'cold') cold, SUM(tier = 'hot') hot, COUNT(*) total "
            "FROM memories GROUP BY project ORDER BY total DESC"),
        "by_agent": rows(
            "SELECT COALESCE(agent, '(없음)') agent, "
            "SUM(tier = 'cold') cold, SUM(tier = 'hot') hot, COUNT(*) total "
            "FROM memories GROUP BY agent ORDER BY total DESC"),
        "by_month": rows(
            "SELECT substr(created_at, 1, 7) month, "
            "SUM(tier = 'cold') cold, SUM(tier = 'hot') hot "
            "FROM memories GROUP BY month ORDER BY month"),
        "sessions": rows(
            "SELECT key, value FROM meta WHERE substr(key, 1, 14) = 'active_session' ORDER BY key"),
    }


def _filters(q: dict) -> tuple[str, list]:
    clauses, params = [], []
    for column in ("project", "tier", "agent"):
        value = q.get(column, "")
        if value == "(없음)":
            clauses.append(f"{column} IS NULL")
        elif value:
            clauses.append(f"{column} = ?")
            params.append(value)
    if q.get("superseded") == "1":
        clauses.append("superseded_by IS NOT NULL")
    if q.get("q"):
        # instr, not LIKE: `%` and `_` in a query are text, not wildcards.
        clauses.append("(instr(topic, ?) > 0 OR instr(content, ?) > 0)")
        params += [q["q"], q["q"]]
    if q.get("entity"):
        clauses.append("id IN (SELECT memory_id FROM memory_entities WHERE entity = ?)")
        params.append(q["entity"])
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def memories(con, q: dict) -> dict:
    where, params = _filters(q)
    offset = max(int(q.get("offset") or 0), 0)
    items = con.execute(
        "SELECT id, topic, tier, project, agent, created_at, superseded_by IS NOT NULL superseded, "
        "substr(content, 1, 200) snippet, "
        "(SELECT COUNT(*) FROM memory_entities me WHERE me.memory_id = m.id) entities "
        f"FROM memories m{where} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
        params + [PAGE_SIZE, offset]).fetchall()
    return {"total": _count(con, f"SELECT COUNT(*) FROM memories{where}", params),
            "offset": offset, "page": PAGE_SIZE, "items": [dict(r) for r in items]}


def memory(con, memory_id: str) -> dict | None:
    row = con.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["entities"] = [dict(r) for r in con.execute(
        "SELECT e.display, e.type FROM memory_entities me JOIN entities e ON e.name = me.entity "
        "WHERE me.memory_id = ? ORDER BY e.display", (memory_id,))]
    out["relations"] = [dict(r) for r in con.execute(
        "SELECT source, predicate, target FROM relations WHERE memory_id = ? "
        "ORDER BY source, predicate, target", (memory_id,))]
    out["chunks"] = _count(con, "SELECT COUNT(*) FROM memory_chunks WHERE memory_id = ?",
                           (memory_id,))
    out["supersession"] = store._supersession(con, row)
    out["corrects"] = [
        {"id": r["id"], "topic": r["topic"]}
        for r in con.execute("SELECT * FROM memories WHERE superseded_by IS NOT NULL")
        if (t := store.resolve_ref(con, r["superseded_by"])) is not None and t["id"] == memory_id
    ]
    return out


def recall(con, q: dict) -> dict:
    """`store.recall()` exactly as `mem_recall` calls it, defaults included (limit 5, detail 1).

    `rerank` is the tool's three states: auto (the gate decides), on (forced past the
    gate), off. Always traced -- tracing only records (tests/test_recall_trace.py).
    """
    started = time.perf_counter()
    out = store.recall(
        con, q.get("query", ""), limit=min(max(int(q.get("limit") or 5), 1), 50),
        project=q.get("project") or None, tier=q.get("tier") or None,
        agent=q.get("agent") or None,
        rerank={"on": True, "off": False}.get(q.get("rerank", "")), trace=True)
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return out


def _candidates(con, target: dict) -> int:
    """How many memories a drop's target could mean, to tell an ambiguous target from one
    that is simply gone. The same tests the resolvers apply: file name + heading for hot
    (`collect_queries._moved`), topic prefix for cold (`eval_paraphrase.resolve`)."""
    if target.get("tier") == "hot":
        name = Path(str(target.get("path") or "")).name
        return _count(con, "SELECT COUNT(*) FROM memories WHERE tier = 'hot' "
                           "AND substr(path, length(path) - ? + 1) = ? AND topic = ?",
                      (len(name) + 1, f"/{name}", str(target.get("heading") or "")[:120]))
    prefix = str(target.get("topic_match") or "")
    return _count(con, "SELECT COUNT(*) FROM memories WHERE substr(topic, 1, ?) = ?",
                  (len(prefix), prefix)) if prefix else 0


def _malformed(drop: dict) -> str:
    """What `collect_queries._load` lets through but a replay cannot run: it checks
    `str(query)`, so a JSON null passes as the text "None"."""
    if not isinstance(drop.get("query"), str) or not drop["query"].strip():
        return "query 가 문자열이 아니다"
    if drop.get("limit") is not None and not isinstance(drop["limit"], int):
        return "limit 이 정수가 아니다"
    if drop.get("rerank") not in (None, True, False):
        return "rerank 가 true/false/null 이 아니다"
    return ""


def _replay(con, drop: dict, memory_id: str, limit: int, rerank) -> dict:
    out = store.recall(con, drop["query"], limit=limit, rerank=rerank, trace=True)
    ids = [r["id"] for r in out["results"]]
    fused = out.get("trace", {}).get("fused", [])
    return {"limit": limit, "rerank": rerank,
            "rank": ids.index(memory_id) + 1 if memory_id in ids else None,
            "fused_rank": fused.index(memory_id) + 1 if memory_id in fused else None,
            "fused_window": len(fused)}


def recall_log(con) -> dict:
    """Every drop in recall-log/, its target resolved, replayed twice: with the window and
    rerank flag it was asked with, and with today's defaults. The two are kept apart --
    the drops were asked with limits of 4, 5 and 6 and mixed rerank flags, so one replay
    under one setting would compare different questions.

    A miss is reported as-is, never retried with a wider window to find the target:
    widening changes what tier slots and the rescue do to the window.
    """
    rows = []
    for path, drop, complaint in collect_queries._load(Path(config.HOME) / "recall-log"):
        item = {"file": path.name, "query": drop.get("query"), "project": drop.get("project"),
                "date": drop.get("date"), "note": drop.get("note"), "target": drop.get("target"),
                "recorded": {k: drop.get(k) for k in ("rank", "limit", "rerank")}}
        complaint = complaint or _malformed(drop)
        if complaint:
            item["problem"] = complaint
            rows.append(item)
            continue
        memory_id, why = collect_queries._resolve(con, drop["target"])
        if memory_id is None:
            many = _candidates(con, drop["target"]) > 1
            item["problem"] = "정답 후보가 여럿 — 모호" if many else f"정답을 못 찾음 ({why})"
            rows.append(item)
            continue
        item["memory"] = {"id": memory_id, "topic": con.execute(
            "SELECT topic FROM memories WHERE id = ?", (memory_id,)).fetchone()["topic"]}
        recorded_limit = drop.get("limit") if isinstance(drop.get("limit"), int) else 5
        item["as_recorded"] = _replay(con, drop, memory_id, recorded_limit, drop.get("rerank"))
        item["current"] = _replay(con, drop, memory_id, 5, None)
        rows.append(item)
    return {"drops": rows}


# `entities.type` is free-form (57 spellings on 2026-10-03, pr / pull_request / "pull request"
# among them). Grouped here for colour only; the store keeps what the writer said.
GROUPS = {
    "개념": {"concept", "pattern", "mechanism", "knowledge", "design", "constraint", "policy",
             "contract", "gate"},
    "오류": {"error", "defect", "test_defect", "finding", "bug-fix", "protocol-fix"},
    "결정": {"decision", "recommendation", "proposal", "outcome", "assessment", "review_verdict"},
    "도구": {"tool", "hook", "harness", "component", "configuration", "product", "file", "test",
             "validation"},
    "프로젝트·문서": {"project", "person", "reference", "canonical reference", "document", "deck",
                  "artifact", "memory", "draft", "review_draft"},
    "작업 항목": {"pr", "pull_request", "pull-request", "pull request", "commit", "issue", "review",
              "change", "feature", "ci-feature", "version", "event"},
}


SPOKES = 4


def group_of(kind: str) -> str:
    for group, kinds in GROUPS.items():
        if kind in kinds:
            return group
    return "작업 항목" if kind.startswith("release") else "기타"


def graph(con, q: dict) -> dict:
    """The relation graph, cut to `limit` entities: hubs by distinct neighbours, each with
    its `SPOKES` best-connected neighbours.

    Not simply the top `limit` by degree. This graph is nearly a forest (3,819 entities,
    3,810 edges on 2026-10-03), so hubs rarely touch each other: the plain cut at 200
    left 44 isolated dots in 55 pieces. Hub-plus-4 keeps the same 45 hubs with 0 isolated
    in 5 pieces; more spokes per hub crowd hubs out (8 -> 37 hubs, all -> 17).

    Distinct neighbours, not relation rows: one pair stated by several memories, or under
    several predicates, is several rows, and counting rows reorders the top of this store
    (PR #1157 is 45 rows but 41 neighbours). Edge weight keeps the row count instead.
    """
    limit = min(max(int(q.get("limit") or 200), 10), 1000)
    hide = set(filter(None, q.get("hide", "").split(",")))
    where, params = _filters({k: q.get(k, "") for k in ("project", "tier", "agent")})
    entities = {r["name"]: r for r in con.execute("SELECT name, display, type FROM entities")}

    weight: Counter = Counter()
    endpoints: set[str] = set()
    for r in con.execute(
        "SELECT r.source, r.target, COUNT(*) n FROM relations r "
        f"JOIN memories m ON m.id = r.memory_id{where} GROUP BY r.source, r.target", params):
        a, b = sorted((r["source"], r["target"]))
        if a == b:
            continue
        endpoints |= {a, b}
        if not hide & {group_of(entities[a]["type"]), group_of(entities[b]["type"])}:
            weight[(a, b)] += r["n"]
    neighbours: dict[str, set] = {}
    for a, b in weight:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    order = sorted(neighbours, key=lambda n: (-len(neighbours[n]), n))
    top: list[str] = []
    for hub in order:
        spokes = sorted(neighbours[hub], key=lambda n: (-len(neighbours[n]), n))[:SPOKES]
        top += [n for n in [hub, *spokes] if n not in top][:limit - len(top)]
        if len(top) >= limit:
            break
    keep = set(top)
    # Counted before hiding, so a hidden group keeps its legend entry and can be shown again.
    groups = Counter(group_of(entities[n]["type"]) for n in endpoints)
    return {
        "nodes": [{"name": n, "display": entities[n]["display"], "type": entities[n]["type"],
                   "group": group_of(entities[n]["type"]), "degree": len(neighbours[n])}
                  for n in top],
        "edges": [{"source": a, "target": b, "weight": w}
                  for (a, b), w in weight.items() if a in keep and b in keep],
        "total_nodes": len(neighbours),
        "total_edges": len(weight),
        "groups": [{"name": g, "count": groups.get(g, 0), "hidden": g in hide}
                   for g in [*GROUPS, "기타"]],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — http.server's naming
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        routes = {
            "/api/status": lambda con: status(con),
            "/api/overview": lambda con: overview(con),
            "/api/memories": lambda con: memories(con, q),
            "/api/memory": lambda con: memory(con, q.get("id", "")),
            "/api/graph": lambda con: graph(con, q),
            "/api/recall": lambda con: recall(con, q),
            "/api/recall-log": lambda con: recall_log(con),
        }
        if url.path not in routes:
            return self._send(404, b"not found", "text/plain")
        con = connect_ro()
        try:
            body = routes[url.path](con)
        finally:
            con.close()
        if body is None:
            return self._send(404, b"not found", "text/plain")
        self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")

    def _send(self, code: int, body: bytes, kind: str):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 — the base signature
        pass


def main():
    parser = argparse.ArgumentParser(description="Read-only dashboard over the store.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # The switch `AGENT_MEMORY_HOT=0` sets, flipped for this process only so recall()
    # searches the mirror as it stands instead of re-indexing it. In main(), not at import:
    # importing this module must not turn sync off for whoever imported it. Should anything
    # reach sync anyway, the read-only connection refuses the write rather than performing it.
    config.HOT_ENABLED = False
    connect_ro().close()  # fail now, not on the first request, if there is no store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}  ({config.DB_PATH}, read-only)")
    server.serve_forever()


if __name__ == "__main__":
    main()
