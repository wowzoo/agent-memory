# agent-memory

한국어 우선 에이전트 메모리 MCP 서버. 로컬 SQLite, Kiwi 형태소 렉시컬 검색, 한국어 파인튜닝 임베딩.

[MARM](https://github.com/Lyellr88/marm-memory)을 베이스로 재설계했습니다. MARM은 잘 만들어진
프로젝트지만 검색 계층이 영어를 전제로 하고 있어서, 한국어 기억에 대해서는 저장은 되고 되찾기가
안 되는 상태였습니다.

같은 부류의 도구들과 이 저장소의 위치는 [`docs/market-survey-2026-09.md`](docs/market-survey-2026-09.md)
에 정리했습니다(2026-09-26 조사).

## 왜 다시 만들었나 — 실측

MARM에 실제로 쌓인 기억 99건(**95건이 한국어**)으로 측정했습니다.

**1. 임베딩 모델이 영어 전용이었다.** `jinaai/jina-embeddings-v2-small-en` 하드코딩
(`config/settings.py:198`, env 오버라이드 없음). 한국어에서는 유사·무관 구분이 사라집니다.

| | 유사쌍 | 무관쌍 | 분리 마진 |
|---|---|---|---|
| 한국어 | 0.978 | 0.971 | **0.007** |
| 영어 (대조) | 0.885 | 0.629 | 0.256 |

known-item 검색으로 재보면 (동일 청크 400토큰/80중첩, 95건):

| 모델 | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| `jina-v2-small-en` (MARM) | 31.6% | 65.3% | 0.462 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 29.5% | 53.7% | 0.415 |
| **`multilingual-e5-small-ko`** | **71.6%** | **93.7%** | **0.820** |

다국어 범용 모델로는 부족하고, 한국어 파인튜닝 모델이어야 했습니다.

**2. FTS5 토크나이저가 한국어 조사를 못 뗀다.** MARM은 `tokenize='porter ascii'`
(`core/memory_db.py:306`). LIKE 부분문자열을 정답으로 두고 비교하면:

| 질의 | 실제 포함 | `porter ascii` | `trigram` | **Kiwi 형태소** |
|---|---|---|---|---|
| 패치 | 33 | 21 (64%) | 0 (0%) | **33 (100%)** |
| 검증 | 75 | 56 (75%) | 0 (0%) | **75 (100%)** |
| 커밋 | 80 | 77 (96%) | 0 (0%) | **79 (99%)** |
| 메모리 | 20 | 15 (75%) | 20 (100%) | 19 (95%) |

`trigram`은 흔한 대안이지만 **3글자 미만 질의를 못 찾습니다.** 2음절 단어가 한국어의 기본
단위이므로 탈락입니다. 그래서 Kiwi로 직접 형태소 분석해 정규화된 토큰 스트림을 FTS5에 넣습니다.

**3. 개념 그래프는 이식이 불가능했다.** MARM의 엔티티 추출은 spaCy `noun_chunks`가 입구인데
**spaCy 한국어에는 `noun_chunks`가 없습니다** (`KoreanDefaults.syntax_iterators == {}`).
술어 분류도 영어 키워드(`"decided"`, `"fix"`, `"implement"`)와 영어 의존구문 LCA에 묶여 있습니다.
이 코퍼스 실측 결과는 엔티티 361 / 관계 0이었습니다.
→ 추출을 버리고 에이전트가 써서 넘기는 방식으로 새로 만들었습니다 (아래 "개념 그래프").

## hot / cold 두 계층

MARM은 자기 DB만 봤습니다. 하지만 이 환경에서 **권위 있는 메모리는
`~/.claude/projects/<slug>/memory/` 아래의 마크다운 파일**입니다 — 매 세션 컨텍스트에 로드되고
손으로 편집되니 그쪽이 SSOT이고 DB는 아카이브입니다. 아카이브만 검색하면 파일이 이미 정정한
사실의 낡은 사본으로 답하게 됩니다.

그래서 hot 파일을 같은 테이블에 색인하고 `tier`로 표시합니다. 결과마다 `tier`와(hot이면) `path`가
붙습니다. 파일이 바뀌면 다음 `mem_recall` 때 자동 재색인되고, 해시가 같으면 건너뜁니다 — 66개
파일 최초 색인 67초, 이후 무변경 확인 **0.18초**.

재색인 단위는 파일이 아니라 **섹션**입니다. hot memory_id 는 `uuid5(path|heading|sha256(body))`
라서, 문단 하나를 고치면 그 섹션만 새 id 가 되고 나머지는 행·벡터·그래프 주석까지 그대로
남습니다. 두 가지를 동시에 해결합니다 — ① `memory_entities`·`relations` 는 `memory_id` 에
CASCADE 로 매달려 있어서, 파일 단위로 지우고 다시 넣으면 손대지도 않은 섹션의 주석이 함께
죽습니다(주석 하나는 기억 전문을 읽어야 나옵니다) ② 편집된 섹션은 id 가 바뀌므로 옛 주석이
새 본문에 **재사용되지 않습니다** — 거짓 주석이 원리적으로 불가능합니다. 위치는 키에 넣지
않습니다: 넣었을 때는 섹션을 위에 하나 끼우면 아래 전부가 번호가 밀려 무관하게 무효화됐습니다.
모델을 갈아서 전량 재임베딩이 필요할 때는 `hot.sync(con, force=True)` 가 그대로 재구축합니다.

⚠️ **한 스토어를 여러 프로젝트가 공유합니다.** MCP 는 프로젝트 단위로 등록하지만 DB 는
`~/.agent-memory/memory.db` 하나입니다. 그래서 각 서버 프로세스는 자기 프로젝트의 memory 디렉터리만
보고, `hot_files` 에 남은 **다른 프로젝트의 행은 "사라진 파일"처럼 보입니다.** 그것까지 지우면
그 기억의 그래프 주석이 `memory_id` CASCADE 로 함께 죽고, 재색인해도 주석은 돌아오지 않습니다 —
두 프로젝트를 번갈아 쓰면 recall 마다 서로의 주석 작업을 파괴했습니다(실측). 그래서 삭제 판정을
**현재 `hot_dirs()` 밑의 경로로만** 한정합니다. 대가는 hot 디렉터리 자체가 없어졌을 때 행이
남는 것이고, 손으로 쓴 주석을 지우는 것보다 훨씬 쌉니다(파일이 돌아오면 덮어씁니다).
`tests/test_hot_reindex.py` 가 이걸 지킵니다. 부수적으로, 프로젝트 A 의 recall 은 프로젝트 B 의
hot 섹션도 후보로 봅니다 — cold 가 이미 그런 것과 같은 설계이고, 결과의 `path` 로 구분됩니다.

⚠️ **같은 이유로 `project` 도 cwd 에서 오면 안 됩니다.** `detect_project()` 는 현재 디렉터리
이름을 쓰는데, 그건 자기 디렉터리를 미러링하는 보통의 경우에만 맞습니다. `AGENT_MEMORY_HOT_PATHS`
로 남의 memory 디렉터리를 가리킨 순간 **그 프로젝트의 섹션이 이쪽 이름으로 다시 라벨됩니다** —
실측했고, `project=` 필터와 `mem_recall(project=...)` 이 조용히 틀려집니다. 그래서 디렉터리를
처음 색인한 프로젝트 이름을 `hot_files.project` 에 적어 두고 이후 색인은 그것을 물려받습니다
(schema v6). 기존 스토어는 `db._migrate` 가 컬럼을 추가하고 그 파일이 만든 `memories.project`
에서 백필합니다.

순위 규칙을 정하는 데 두 가지를 시도했다가 버렸습니다. 둘 다 실측입니다 (hot 582섹션 vs
cold 101건, 파일이 6배 많음):

| 방식 | hybrid R@1 | R@5 | MRR |
|---|---|---|---|
| hot에 일괄 가점 1.25배 | 0% | 40% | 0.191 |
| 티어 교대 배치(interleave) | 0% | 80% | 0.394 |
| **관련도 정렬 + 티어 최소 슬롯** | **35%** | **75%** | **0.520** |
| (참고) cold 단독 = 엔진 성능 | 55% | 90% | 0.692 |

일괄 가점은 파일이 답하지 않는 질문에서도 파일을 올려서 아카이브를 도달 불가로 만들었습니다.
교대 배치는 1번 슬롯을 항상 hot에 써서 R@1이 0에 고정됐습니다. 남은 답은 **관련도만으로
정렬하고, 어느 한쪽이 아예 사라지지 않게 슬롯을 보장**하는 것입니다. 순서는 관련도가
정하고, 이 장치는 "사라지지 않음"만 정합니다. **몇 칸을 보장할지는 2026-08-30 에 처음
실측했습니다 — 기본값은 `1` 입니다**(⇒ [`docs/measurements.md`](docs/measurements.md)
`### 티어 슬롯은 한 칸이면 됩니다`).

실제 동작:

```
그래프 기능은 결국 설치하기로 했었나  → [hot, hot, cold, hot, cold]
   [hot ] project_marm_memory_install.md  graph 기능 도입 판단 (2026-07-23 …)
한국어 검색이 왜 안 됐지            → [cold, hot, hot, hot, cold]
   [cold] old-new-compare-v2-chronicle    2026-08-04 marm 한국어 concept graph 실측 …
```

첫 질문은 파일의 정제된 판단이 이기고, 둘째는 아카이브의 장문 실측이 이깁니다 — 규칙이 아니라
관련도가 정한 결과입니다. `tier="hot"` / `"cold"`로 한쪽만 볼 수도 있습니다.

hot 계층은 이 도구에서 **읽기 전용**입니다. `mem_write`는 파일에 절대 쓰지 않습니다.

## 누가 썼는지 — `agent`

claude · codex · kiro 를 전역 등록하면 **한 DB 를 세 하네스가 같이 씁니다**. 결과에 `agent` 가
붙습니다 — `'claude'` / `'codex'` / `'kiro'` — 그리고 `mem_recall(agent=...)` 로 한쪽만 볼 수
있습니다. `source` 를 겸용하지 않은 이유는 그게 이미 **유입 경로**(`write` / `hot` /
`marm:*`)를 뜻하기 때문입니다. 작성자는 별개의 사실입니다.

이름은 두 곳에서 옵니다. 순서가 있습니다.

1. `AGENT_MEMORY_AGENT` — 각 하네스가 자기 MCP 등록의 env 블록에 적습니다. 운영자가 통제하는
   유일한 값이라 **이쪽이 이깁니다**.
2. MCP `initialize` 의 `clientInfo.name` — 설정이 없어도 뭔가 찍힙니다. 다만 스펙상 optional
   이고(2026-07-28), 두 하네스가 같은 문자열을 보내도 막을 방법이 없습니다.

둘 다 없으면 `agent: null` 입니다. 추측하지 않고, `mem_write` 응답에 그대로 보여 줍니다 —
env 가 안 꽂혔다는 사실 자체가 봐야 하는 정보입니다. 표기는 정규화합니다(`Claude Code` →
`claude-code`), 그래야 필터 키가 한 개로 고정됩니다.

**순위에는 넣지 않습니다.** `tier` 와 `superseded_by` 가 그런 것과 같습니다. 이 코퍼스에서
관련도 아닌 출처로 순서를 바꾸려 한 시도가 셋 있었고(hot 일괄 가점 · hot 문맥 접두 ·
확신 가중 β), 셋 다 산 것보다 판 게 많았습니다. `agent` 는 표시와 필터입니다.

⚠️ **hot 행의 `agent` 는 미러링한 프로세스가 아니라 디렉터리 주인입니다.** `hot_files.agent`
에 처음 색인한 쪽이 적고 이후는 물려받습니다 — `project` 가 schema v6 에서 당한 것과 같은
함정이고(남의 memory 디렉터리를 가리키면 그쪽 섹션이 이쪽 이름으로 다시 라벨됐습니다),
Codex 의 recall 이 Claude 의 memory 파일을 미러링한다고 그게 Codex 것이 되지는 않습니다.
단 `~/.claude/projects/` 밑은 물려받기 전에 **위치로 `claude`** 입니다 — 물려받기는 누군가
이름을 적은 뒤에야 작동해서, 09-07 이관 후 새 경로를 Codex 가 먼저 색인한 디렉터리 넷(36파일)이
`codex`·`codex-mcp-client` 로 찍히고 이후 파일이 전부 그걸 물려받았습니다(09-26 발견·정정).

⚠️ **활성 세션도 agent 별로 갈랐습니다.** `meta.active_session` 한 행을 세 하네스가 공유하면
Codex 가 세션을 지정한 순간 Claude 의 다음 무명 write 가 그 버킷으로 들어갑니다. 이제 키가
`active_session:<agent>` 이고, 옛 전역 키는 **폴백으로만** 읽습니다 — 이관 전 스토어가 각
agent 의 첫 write 까지 세션을 유지합니다.

`agent` 가 없던 시절의 cold 기억은 `null` 입니다. 세 하네스가 이미 같이 쓰고 있었으니 사후
귀속이 불가능하고, 지어내는 대신 비워 뒀습니다(실측: cold 343건). hot 은 다릅니다 — 옛 hot
디렉터리는 전부 `~/.claude/projects/` 아래였으므로(300/300 확인) 마이그레이션이 `'claude'` 로
채웁니다. 그래서 `agent=` 필터는 hot 전체를 첫 recall 부터 덮습니다.

## 지금 성능

🔴 **최신 측정은 2026-09-26, 출하 설정입니다** (게이트 리랭크 · `limit=5` · 오늘 기록을 뺀 사본):
cold **R@1 61.1 / R@5 75.0** · hot 38질의 **R@1 42.1 / R@5 73.7** · 축자 60건 **R@1 78.3 / R@5 96.7**.

그 전 기준선은 2026-09-13 (cold 439 · hot 1,620 · `limit=5` · 기본 경로):
cold **R@1 59.7 / R@5 84.7** · hot 38질의 **R@1 36.8 / R@5 73.7** · 축자 60건 **R@1 51.7 / R@5 93.3**.
레인 단독은 cold 에서 lexical 47.2/70.8 · semantic 50.0/70.8 이고, 융합이 두 쪽 다 이깁니다.
경로가 달라 두 줄을 바로 비교하면 안 됩니다.

**조건·근거·기각 이력은 전부 [`docs/measurements.md`](docs/measurements.md) 에 있습니다** — 16개
절이 리랭커 게이트 · 확신 가중 β · hot 브레드크럼 · 명사 결합 · 구제 깊이 · 티어 슬롯 · `RRF_K` ·
후보 풀을 각각 어떤 상태에서 어떤 수치로 채택하거나 기각했는지, 그리고 잘못 재서 판정을 두 번
뒤집은 기록까지 답니다. 여기에 요약을 두지 않는 이유는 그 문서가 세 번 확인한 사실입니다 —
**조건을 뗀 수치는 근거가 아닙니다.**

실험을 다시 열기 전에 그 문서에서 먼저 읽을 넷:

- `### 🔴 두 가지 측정 위생` — 이 저장소의 기록이 이 저장소의 픽스처를 오염시킵니다.
- `### 🔴 세 번째 위생 — 프로젝트가 이사하면 hot 미러가 두 벌 남습니다` — 실측 43% 가 유령이었습니다.
- `### 정리 후 기준선` — 위 수치의 출처이고, 게이트가 R@1 을 20%p 깎는다는 측정입니다.
- `### 리랭커는 값을 하지만 3초를 받습니다` — 그런데도 기본 on 인 이유(비용 장치입니다).

## 이전하면서 발견한 MARM 데이터 결함

원본은 읽기 전용으로만 열고, 교정은 이전 과정에서만 합니다.

- **본문 HTML 이스케이프** — `memories` 99건 중 **92건**이 `&quot;`로 저장돼 있고
  `log_entries` 97건은 전부 원문입니다. 본문 손상이자 색인 오염(`quot` 토큰)이며, 동시에 같은
  엔트리의 두 사본이 문자열로 달라져 중복 제거를 통과시켰습니다. `html.unescape()` 후 중복
  스킵이 83 → **95건**으로 늘어 근중복 12건이 사라졌습니다.
- **`log_entries.topic`이 제목이 아님** — 97건 중 84건이 `"general"`입니다. 여기서는 topic이
  3배 가중되고 독립 벡터를 가지므로 그대로 쓰면 해가 됩니다. 본문 첫 줄에서 뽑습니다.
- **`session_name`에 본문이 들어간 건 2건** — 각각 1,529자·1,232자. MARM의 `"Session: [name]"`
  접두사 파싱 사고로 보입니다. 첫 줄이 의도된 이름이어서 복구했습니다.

## 왜 툴이 3개뿐인가

Claude Code 세션 트랜스크립트 76개를 집계한 실사용량입니다. MARM의 25개 툴에 대해 총 129회 호출:

| 툴 | 호출 | |
|---|---|---|
| `log_entry` | 99 | → `mem_write` |
| `smart_recall` | 11 | → `mem_recall` |
| `log_show` | 10 | → `mem_sessions` |
| `concept_build` / `concept_recall` | 7 | → `mem_graph` (재설계) |
| `graph_index` | 2 | 코드 그래프, 제외 |
| notebook · summary · delete · compaction · code_lookup · graph_trace/impact/architecture | **0** | 제외 |

상위 3개가 94%였습니다. 쓰기:읽기 비율은 99:21 ≈ 5:1 — 저장은 하는데 되찾아 쓰지 못하는
상태였고, 위의 검색 결함이 그 원인으로 보입니다.

## 설치

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

# 임베딩 모델 준비 (HF에는 safetensors만 있어 ONNX는 로컬에서 만듭니다)
.venv/bin/pip install -e '.[export]'
.venv/bin/python scripts/export_model.py
```

⚠️ 이미 기억을 쌓은 스토어의 모델 디렉터리를 **다른** 아티팩트로 덮으면 저장된 청크 벡터와
어긋납니다. `AGENT_MEMORY_MODEL_NAME` 이 그대로면 `mem_recall` 의 모델 불일치 warning 도 뜨지
않으므로 조용히 나빠집니다(2026-08-09 실측: 교체 전후 프로브 cosine 0.936~0.951). 교체할 때는
청크를 전부 다시 임베딩하고, 실행 중인 MCP 서버를 재시작하십시오 — 서버는 모델을 프로세스에
한 번만 올립니다.

리랭커는 선택입니다. 안 만들면 `rerank=True` 를 줘도 조용히 베이스 순서로 답합니다 —
호출자가 원한 건 더 나은 순서이고, 요청이 실패하는 것보다 베이스 순서라도 답하는 편이 낫습니다.

```bash
.venv/bin/python scripts/export_reranker.py \
    --repo Dongjin-kr/ko-reranker --out ~/.agent-memory/models/ko-reranker
.venv/bin/python - <<'EOF'
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
src = Path.home() / ".agent-memory/models/ko-reranker"
dst = Path.home() / ".agent-memory/models/ko-reranker-int8"
dst.mkdir(exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
    (dst / f).write_bytes((src / f).read_bytes())
quantize_dynamic(src / "model.onnx", dst / "model.onnx", weight_type=QuantType.QInt8)
EOF
```

fp32 export(2.1GB)는 양자화의 입력으로만 필요하니 int8(552MB)을 확인한 뒤 지워도 됩니다.

MCP 등록 (Claude Code):

```bash
claude mcp add agent-memory -- /절대경로/agent-memory/.venv/bin/agent-memory
```

MARM 데이터 이전:

```bash
.venv/bin/agent-memory-migrate            # 기본 소스: ~/.marm/marm_memory.db (읽기 전용)
```

`memories`와 `log_entries`는 MARM이 같은 본문을 양쪽에 쓰기 때문에 대부분 중복입니다.
`memories`를 기준 레코드로 삼고, 본문이 이미 표현되지 않은 로그만 추가로 이전합니다.

## 개념 그래프 — 추측하지 않고 받아 적습니다

MARM의 개념 그래프는 이 코퍼스에서 실측했을 때 **엔티티 361개 / 관계 0개**를 냈고, 경계가
조사 안쪽에서 잘렸습니다(`"결론은 marm-memory"`, `"캡처)는"`). `en_core_web_sm`이 은/는/가/를을
못 떼기 때문입니다. spaCy 한국어 모델로 바꿔도 해결되지 않습니다 — `noun_chunks`가 아예 없고
(`syntax_iterators == {}`), 그게 그 파이프라인의 입구입니다.

그래서 추출하지 않습니다. **기억을 이미 컨텍스트에 들고 있는 에이전트가 엔티티와 관계를 써서
넘깁니다.** 형태소 경계를 추측할 일이 없고, 술어도 동시출현(`co_occurs_with`) 대신 실제 의미를
씁니다.

```
mem_write(content="...", 
          entities=[{"name": "claim-sources sensor", "type": "tool"},
                    {"name": "이슈 #665", "type": "error"}],
          relations=[{"source": "claim-sources sensor", "predicate": "causes",
                      "target": "이슈 #665"}])
```

이미 있는 기억(MARM에서 넘어온 것 포함)에는 `mem_annotate(memory_id, entities, relations)`로
붙입니다. 재주석은 병합이 아니라 **교체**라서 정정이 누적으로 변하지 않습니다.

값이 나오는 지점은 **기억을 건너뛰는 경로**입니다. 실제 주석한 4건에서:

```
mem_graph("AI-DLC 2.5.11", hops=2)
  AI-DLC 2.5.11        --adds        --> claim-sources sensor      [기억 A]
  claim-sources sensor --causes      --> t66/t89 테스트 실패          [기억 A]
  claim-sources sensor --causes      --> 이슈 #665                  [기억 B]
  claim-sources sensor --invalidates --> designer-export 골든 fixture [기억 A]
```

`2.5.11 → claim-sources sensor → 이슈 #665`는 서로 다른 두 기억에 걸친 경로로, 어느 한 기억을
읽어도 보이지 않습니다. 관계마다 `memory_id`·`memory_topic`·`tier`가 붙어 근거를 바로 읽을 수
있습니다.

`mem_graph()`를 인자 없이 부르면 그래프의 형태를 봅니다 — 총계, 차수 기준 허브,
**isolated**(기억에는 기록됐지만 무엇과도 연결되지 않은 엔티티 = 붙잡아 놓고 잇지 않은 사실),
술어 분포. 엔티티 이름 매칭은 대소문자·공백을 무시하고, 못 찾으면 `did_you_mean`을 돌려줍니다.

관계는 근거 기억이 삭제되면 함께 사라지고, 아무 기억도 참조하지 않는 엔티티는 정리됩니다.

### 주석은 세션마다 다시 붙여야 합니다 (`scripts/annotate.py`)

hot 섹션은 본문을 고치면 해시가 바뀌어 새 `memory_id`가 되고, **설계상 주석을 잃습니다**
(거짓 주석을 원리적으로 막는 대가입니다). 메모리 파일을 건드리는 세션마다 스토어가 미주석
쪽으로 흘러가고, 주석이 recall 의 그래프 레인에 들어가므로 그 흐름이 회수 품질 문제가 됩니다.

```
python scripts/annotate.py --list                  # 주석 없는 기억
python scripts/annotate.py batch.json --dry-run    # 해석만 확인
python scripts/annotate.py batch.json
```

`mem_annotate` 인자가 아니라 **파일로 넣습니다.** MCP 툴 인자로 보낸 한글은 3분의 1가량이
조용히 깨지고(8배치 실측: 엔티티 272개 중 94개), 파일 왕복은 보존합니다. 배치는
`{"memory_id": "3f68be58", "entities": [...], "relations": [...]}` 목록이고 id 접두사로
해석합니다. `MEMORY.md`의 색인 섹션은 `--list`에서 빼 둡니다 — 다른 기억을 가리키는 포인터뿐이라
주석하면 그 엔티티만 중복됩니다.

⚠️ **`--dry-run`은 쓰고 나서 되돌리는 방식이 아닙니다.** `graph.attach`가 스스로 커밋해서 되돌릴
트랜잭션이 남지 않으므로, 해석과 건수만 확인하고 attach 는 부르지 않습니다.

🔴 **주석 부착은 그 기억의 주석 집합을 통째로 교체합니다.** 엔티티 하나만 담은 배치를 넣으면
7개였던 기억이 1개가 됩니다 — 다른 프로젝트 세션이 실제로 당했습니다. 그래서 출력이 **바뀌기 전
값을 함께** 찍고, 줄어들면 `⚠ fewer than before` 를 붙입니다. 더하려면 `--append` 로 기존 집합과
병합합니다(손으로 전체를 다시 싣는 대신).

```
89c60bd6    1e   0r  (was 7e 0r)  ⚠ fewer than before   ← 교체(기본)
89c60bd6    8e   0r  (was 7e 0r)                        ← --append
```

⚠️ **이 DB 를 백업할 때 `cp` 를 쓰지 마십시오.** WAL 모드라 로그가 본체보다 몇 MB 뒤처져 있고
(실측: `memory.db` 14MB ↔ `memory.db-wal` 9.6MB), `cp` 사본은 최근 기록을 빠뜨립니다.
`sqlite3 <db> ".backup <out>"` 로 뜨고 `integrity_check` + 행수로 확인하십시오.
`scripts/fix_entity_names.py` 의 자동 백업도 이 방식으로 고쳤습니다.

⚠️ **주석을 늘리면 회수가 좋아진다고 가정하지 마십시오.** [`docs/measurements.md`](docs/measurements.md)
의 그래프 레인 절이 내린 판정 이후 편집으로
주석을 잃은 hot 4섹션을 다시 붙이고 **같은 상태에서 앞뒤를 재봤더니**, cold-only 는 완전히
동일했고(72.2% / 94.4% / 0.819) 두 계층 R@5 는 **94.4%→93.1% 로 1건 내려갔습니다**(MRR
0.807→0.804). 원인은 명확합니다 — 새로 주석한 4섹션이 전부 "그래프" 주제라서, 그래프 레인이
그 섹션들을 후보로 올리고 같은 주제의 cold 정답을 5위 밖으로 밀어냅니다(정답 5위→7위, 밀어낸
3·6위가 새로 주석한 두 섹션). 평가셋 정답이 전부 cold 라 이 세트는 hot 이 올라오는 것을 항상
손실로 셉니다만, 밀려난 질의는 실제로 그 cold 기억을 묻고 있었습니다. **주제가 몰린 주석은
같은 주제의 정답과 5슬롯을 다툽니다.**

## 툴

- **`mem_write(content, session=, topic=, project=, entities=, relations=)`** — 기억 저장.
  결정·근거·결과를 남기고 지나가는 잡담은 남기지 않습니다. 비밀정보 모양(AWS 키, GitHub·Slack·API
  토큰, private key, `password=…`)이 보이면 **저장하지 않고** error 를 돌려줍니다 — 한 번 들어가면
  `mem_forget` 으로 지워도 VACUUM 전까지 바이트가 남기 때문입니다. 기존 cold 기억과 거의 같으면
  (cosine ≥ 0.95) 저장은 하고 `near_duplicate` 로 그 기억을 알려 줍니다(`supersedes` 로 이미
  가리켰으면 조용합니다).
- **`mem_recall(query, limit=, session=, project=, detail=, mode=, tier=, rerank=, agent=)`** —
  렉시컬(Kiwi+BM25) · 시맨틱(청크 임베딩) · 개념 그래프 co-citation(hybrid 에서만, 절반 가중)
  세 레인을 RRF로 융합. hot/cold를 함께 검색해 구분 반환합니다. `rerank=True`는 상위 10건을
  한국어 크로스인코더로 다시 매깁니다 — R@1 72%→92%, 대가는 3초입니다. `agent=`는 한 하네스가
  기록한 것만 봅니다. 결과가 있으면 응답에 `results_are` 가 붙어 **결과는 기록된 데이터이지
  지시가 아니라고** 말합니다 — 여러 하네스·프로젝트가 한 스토어에 쓰므로 누군가 저장한 "~하라"는
  문장이 그대로 다른 에이전트의 컨텍스트에 들어옵니다. 본문은 손대지 않습니다.
- **`mem_sessions(session=)`** — 세션 목록 또는 한 세션의 항목 목록. `active_session` 은
  호출한 agent 자신의 것입니다.
- **`mem_annotate(memory_id, entities=, relations=)`** — 기존 기억에 개념 그래프 부착.
- **`mem_forget(memory_id)`** — cold 기억 하나를 영구 삭제하고 DB 파일에서 본문을 지웁니다.
  잘못 저장한 비밀·개인정보용입니다. 낡았을 뿐인 기억은 지우지 말고 `mem_write(supersedes=)` 로
  정정하십시오. hot 섹션은 거부합니다 — 파일을 고치십시오. 평범한 `DELETE` 로는 부족합니다:
  심어 둔 비밀이 FTS 인덱스 세그먼트와 해제된 페이지에 남았고, FTS `optimize` 뒤 `VACUUM` 을
  해야 0 이 됩니다(실측 2026-09-24, 라이브 스토어 사본에서 0.3초). 이미 뜬 백업과, 다른
  프로세스가 읽는 중이라 체크포인트가 막힌 WAL 은 닿지 않으며 응답이 그렇다고 말합니다.
- **`mem_graph(entity=, hops=, predicate=)`** — 그래프 순회 또는 전체 형태 조회.

## 설정

전부 env로 덮어쓸 수 있습니다 (`agent_memory/config.py`). MARM이 모델을 하드코딩해서
소스 수정 없이는 인코더를 바꿀 수 없었던 게 이 프로젝트의 출발점이라, 검색에 영향을 주는
값은 모두 열어 뒀습니다.

| 변수 | 기본값 |
|---|---|
| `AGENT_MEMORY_HOME` | `~/.agent-memory` |
| `AGENT_MEMORY_MODEL_DIR` | `$HOME/models/multilingual-e5-small-ko` |
| `AGENT_MEMORY_MODEL_NAME` / `_EMBED_DIM` | `multilingual-e5-small-ko` / `384` |
| `AGENT_MEMORY_QUERY_PREFIX` / `_PASSAGE_PREFIX` | `query: ` / `passage: ` (e5 계열 필수) |
| `AGENT_MEMORY_CHUNK_TOKENS` / `_CHUNK_OVERLAP` | `400` / `80` |
| `AGENT_MEMORY_MAX_CONTENT_CHARS` | `50000` (`mem_write` 본문 상한. 지금까지 가장 긴 기억은 8,838자) |
| `AGENT_MEMORY_DUPLICATE_SIMILARITY` | `0.95` (`near_duplicate` 경고 임계값. 라이브 cold 663건에서 2쌍, 둘 다 1분 안에 다시 쓴 기억) |
| `AGENT_MEMORY_TOPIC_WEIGHT` / `_BODY_WEIGHT` | `3.0` / `1.0` (bm25 컬럼 가중) |
| `AGENT_MEMORY_RRF_K` / `_CANDIDATE_POOL` | `10` / `50` (k 는 이 코퍼스 크기에 맞춰 60에서 내림) |
| `AGENT_MEMORY_GRAPH_LANE_WEIGHT` / `_GRAPH_LANE_SEEDS` | `0.75` / `3` (`0`이면 그래프 레인 끔) |
| `AGENT_MEMORY_RERANK` | `1` (게이트가 통과시킨 recall 만 리랭킹 — 실측 여섯 번에 한 번, 평균 0.5초. `0`은 완전히 끔) |
| `AGENT_MEMORY_RERANK_GATE` | `1` (1위를 모든 레인이 데려왔으면 리랭커를 건너뛴다. `0`이면 전부 리랭킹) |
| `AGENT_MEMORY_RESCUE_LANE_LEADERS` | `1` (다른 레인이 못 데려온 레인 1위에게 창 마지막 칸을 준다 — hot R@5 +2.6%p, cold 무손실) |
| `AGENT_MEMORY_RERANK_MODEL_DIR` | `$HOME/models/ko-reranker-int8` |
| `AGENT_MEMORY_RERANK_DEPTH` / `_MAX_TOKENS` | `10` / `512` (깊이 20은 상한 100%, 지연 두 배) |
| `AGENT_MEMORY_HOT_PATHS` | 현재 프로젝트의 Claude Code memory 디렉터리 |
| `AGENT_MEMORY_HOT` | `1` (`0`이면 hot 미러링 끔) |
| `AGENT_MEMORY_MIN_TIER_SLOTS` | `1` (`0`이면 순수 관련도 정렬. 리랭크를 끄고 쓰면 `2` 가 맞다 — cold R@5 +2.8%p) |
| `AGENT_MEMORY_HALF_LIFE_DAYS` | `0` (최신성 가중 끔) |
| `AGENT_MEMORY_AGENT` | — (없으면 MCP `clientInfo.name`, 그것도 없으면 `null`. ⇒ `## 누가 썼는지`) |
| `AGENT_MEMORY_EXTRA_STOPWORDS` | — |

임베딩 모델명·차원은 DB `meta` 테이블에 기록됩니다. 모델을 바꾸면 `mem_recall`이 낡은 벡터로
순위를 매기는 대신 렉시컬 레인으로 내려가고 `warning`을 돌려줍니다.

더 큰 한국어 검색 모델(예: `nlpai-lab/KURE-v1`, 1024차원)로 바꾸려면 `MODEL_DIR`을 그 export로
가리키고 `EMBED_DIM`을 맞춘 뒤 재임베딩하면 됩니다.

## 검증

```bash
.venv/bin/python -m pytest tests/ -q                       # 전체 테스트
.venv/bin/python scripts/eval_paraphrase.py                # 튜닝용 (두 계층)
.venv/bin/python scripts/eval_paraphrase.py --tier cold    # 엔진 성능만
.venv/bin/python scripts/eval_recall.py                    # 회귀 감지용
.venv/bin/python scripts/experiment_topic_signal.py        # 토픽 신호 기여도
.venv/bin/python scripts/experiment_r5_priority.py         # 그래프 레인 가중치 · β
.venv/bin/python scripts/annotate.py --list                # 주석이 빠진 기억
.venv/bin/python scripts/experiment_rerank.py --tier cold  # 리랭커 상한 (모델 없이도 됨)
.venv/bin/python scripts/eval_paraphrase.py --tier cold --rerank   # 리랭커 켠 프로덕션 경로
.venv/bin/python scripts/dashboard.py                      # 읽기 전용 대시보드 http://127.0.0.1:8765
```

대시보드는 상태(정합성·hot 신선도) · 회수(단계별 trace, recall-log 재생) · 내용 · 그래프 네 탭입니다.
스토어를 `mode=ro` 로만 열고 hot 미러를 sync 하지 않습니다 — 근거는 `scripts/dashboard.py` 머리말.

`eval_recall.py`는 실제 `store.recall` 경로를 호출합니다. 각 기억의 본문 중간 문장을 질의로
넣고 원본이 몇 위로 돌아오는지 재므로 라벨링이 필요 없습니다.
