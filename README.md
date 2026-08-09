# agent-memory

한국어 우선 에이전트 메모리 MCP 서버. 로컬 SQLite, Kiwi 형태소 렉시컬 검색, 한국어 파인튜닝 임베딩.

[MARM](https://github.com/Lyellr88/marm-memory)을 베이스로 재설계했습니다. MARM은 잘 만들어진
프로젝트지만 검색 계층이 영어를 전제로 하고 있어서, 한국어 기억에 대해서는 저장은 되고 되찾기가
안 되는 상태였습니다.

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
정렬하고, 어느 한쪽이 아예 사라지지 않게 슬롯 2개만 보장**하는 것입니다. 순서는 관련도가
정하고, 이 장치는 "사라지지 않음"만 정합니다.

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

## 지금 성능

손으로 쓴 패러프레이즈 질의 **72개** 기준 (`scripts/eval_paraphrase.py`):

| 레인 | Recall@1 | Recall@5 | MRR | 중위 순위 |
|---|---|---|---|---|
| lexical (Kiwi + BM25) | 56.9% | 83.3% | 0.672 | 1위 |
| semantic (e5-small-ko) | 58.3% | 88.9% | 0.707 | 1위 |
| **hybrid (RRF)** | **68.1%** | **94.4%** | **0.795** | **1위** |

평가셋은 20개에서 72개로 늘렸습니다(2026-08-09). 추가한 52개는 주석이 붙은 cold 기억 하나당
질의 하나이고, 같은 PR·같은 세대·같은 덱처럼 이웃한 기억과 구별되는 지점을 골라 걸었습니다 —
질의가 애초에 답을 특정하지 못하면 실패가 회수 실패인지 질의 결함인지 갈리지 않기 때문입니다.

⚠️ 두 그룹의 난이도가 같지 않습니다. 원래 20개는 hybrid R@1 60.0% / MRR 0.728, 추가한 52개는
**R@1 71.2% / MRR 0.820** 입니다. 추가분은 기억 본문을 읽은 직후에 쓴 질의라 원래 20개보다
어휘를 더 물려받았습니다(가장 노골적인 24개는 회상투로 다시 썼고, 그 전에는 R@1 84.6% 였습니다).
따라서 **절대 수치는 낙관 쪽으로 치우쳐 있습니다.** 이 평가셋의 용도는 절대 성능 주장이 아니라
변경 전후 비교이고, 질의 1건이 1.4%p 라 20개 시절(1건 = 5%p)보다 그 비교를 믿을 수 있습니다.

두 레인을 합쳐야 어느 한쪽보다 낫습니다. 아래 개선 이력의 수치는 전부 **당시의 20질의 기준**이라
위 표(72질의)와 직접 이어 읽으면 안 됩니다. 출발점(단일 FTS 컬럼 · 토픽 벡터 없음 · HTML
이스케이프 잔존)은 hybrid R@1 25% / MRR 0.462였고, 아래 네 가지로 60.0%까지 왔습니다.

- **토픽을 독립 신호로**: 토픽 줄을 별도 벡터로 임베딩 + FTS 토픽 컬럼 분리 후 3배 가중.
  단독으로는 각각 R@1 25%·30%였는데 함께 쓰니 45%로 올랐습니다 (`scripts/experiment_topic_signal.py`).
- **청킹 truncation 버그 수정**: 토크나이저를 인코딩과 공유해서 청킹 입력이 512토큰으로 먼저
  잘렸고, 4,954토큰 기억이 2청크만 만들었습니다. 저장된 본문의 **42.9%만** 색인돼 있었습니다.
  `tests/test_chunking.py`가 재발을 막습니다. 같은 결함이 **아티팩트 쪽에서 한 번 더** 들어옵니다 —
  `save_pretrained` 가 truncation 을 `tokenizer.json` 에 박기 때문에, 로더가 청킹용 인스턴스의
  truncation 을 **명시적으로 끕니다**(안 끄면 이 스토어에서 599청크가 337로 줄어듭니다).
- **ONNX 아티팩트를 충실한 export 로 교체 + 전량 재색인** (2026-08-09): 기존 아티팩트는 풀링된
  `sentence_embedding` 출력을 내면서 PyTorch 레퍼런스와 프로브 cosine 0.94 로 어긋나 있었습니다.
  `scripts/export_model.py` 산출물(레퍼런스 대비 cosine 1.000)로 갈고 599청크를 다시 임베딩했더니
  패러프레이즈 hybrid R@1 50.0%→60.0% · MRR 0.655→0.729, `eval_recall.py` hybrid R@5 94.9%→99.2%.
  (20질의 기준이라 R@1 +10%p 는 2건입니다.)
- **MARM 데이터 품질 결함 3종 교정** (아래).

평가 하니스가 두 개인 이유: `eval_recall.py`는 문서에서 문장을 **그대로** 떼어 질의로 쓰므로
BM25에 현실에 없는 완벽한 어휘 일치를 줍니다(lexical R@5 100%). 회귀 감지용으로는 유용하지만
레인 가중치를 여기에 맞추면 안 됩니다. 튜닝은 `eval_paraphrase.py`로 합니다.

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

## 툴

- **`mem_write(content, session=, topic=, project=, entities=, relations=)`** — 기억 저장.
  결정·근거·결과를 남기고 지나가는 잡담은 남기지 않습니다.
- **`mem_recall(query, limit=, session=, project=, detail=, mode=, tier=)`** — 렉시컬(Kiwi+BM25)과
  시맨틱(청크 임베딩) 두 레인을 RRF로 융합. hot/cold를 함께 검색해 구분 반환합니다.
- **`mem_sessions(session=)`** — 세션 목록 또는 한 세션의 항목 목록.
- **`mem_annotate(memory_id, entities=, relations=)`** — 기존 기억에 개념 그래프 부착.
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
| `AGENT_MEMORY_TOPIC_WEIGHT` / `_BODY_WEIGHT` | `3.0` / `1.0` (bm25 컬럼 가중) |
| `AGENT_MEMORY_HOT_PATHS` | 현재 프로젝트의 Claude Code memory 디렉터리 |
| `AGENT_MEMORY_HOT` | `1` (`0`이면 hot 미러링 끔) |
| `AGENT_MEMORY_MIN_TIER_SLOTS` | `2` (`0`이면 순수 관련도 정렬) |
| `AGENT_MEMORY_HALF_LIFE_DAYS` | `0` (최신성 가중 끔) |
| `AGENT_MEMORY_EXTRA_STOPWORDS` | — |

임베딩 모델명·차원은 DB `meta` 테이블에 기록됩니다. 모델을 바꾸면 `mem_recall`이 낡은 벡터로
순위를 매기는 대신 렉시컬 레인으로 내려가고 `warning`을 돌려줍니다.

더 큰 한국어 검색 모델(예: `nlpai-lab/KURE-v1`, 1024차원)로 바꾸려면 `MODEL_DIR`을 그 export로
가리키고 `EMBED_DIM`을 맞춘 뒤 재임베딩하면 됩니다.

## 검증

```bash
.venv/bin/python -m pytest tests/ -q                       # 11 tests
.venv/bin/python scripts/eval_paraphrase.py                # 튜닝용 (두 계층)
.venv/bin/python scripts/eval_paraphrase.py --tier cold    # 엔진 성능만
.venv/bin/python scripts/eval_recall.py                    # 회귀 감지용
.venv/bin/python scripts/experiment_topic_signal.py        # 토픽 신호 기여도
```

`eval_recall.py`는 실제 `store.recall` 경로를 호출합니다. 각 기억의 본문 중간 문장을 질의로
넣고 원본이 몇 위로 돌아오는지 재므로 라벨링이 필요 없습니다.
