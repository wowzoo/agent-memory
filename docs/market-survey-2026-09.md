# 에이전트 메모리 시장 조사 — 2026-09-26

같은 부류의 도구가 빠르게 늘어나서, 지형과 이 저장소의 위치를 한 번 정리했습니다. 설계 판정의 근거는
[`measurements.md`](measurements.md) 에 있고, 이 문서는 **바깥에서 무엇이 어떻게 만들어지고 평가되는가**만 다룹니다.

## 조사 조건과 한계

- **열린 웹 검색 없이** 1차 출처(README · 공식 문서 · 가격 페이지 · 벤더 블로그 · arXiv 초록)를 직접 열어 확인했습니다.
  검색 API 할당량이 소진됐고 검색엔진은 CAPTCHA 로 막혔습니다. 그래서 **한국어 웹·뉴스는 거의 보지 못했고**,
  한국 스타트업 제품은 확인하지 못했습니다.
- 스타 수는 2026-09-26 GitHub API 값입니다. mem0 · claude-mem · MemPalace · synaptic-memory 는 따로 다시 조회해 맞췄습니다.
- **벤치마크 수치는 거의 전부 벤더가 스스로 낸 값입니다.** 독립 재현은 확인하지 못했습니다.
- 투자 금액은 공식 발표나 기사로 확인한 것만 적었습니다. 금액을 못 찾은 곳은 "미확인"입니다.

## 1. 지형 — 네 갈래

### 독립 메모리 레이어 (앱 개인화)

| 제품 | 구조 / 쓰기 경로 | 배포 · 라이선스 | 투자 | ★ |
|---|---|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 벡터 + BM25 + 엔티티 링크, 유료 티어에 그래프. 2026-04 알고리즘은 쓰기마다 LLM 추출 1회, **add-only**(UPDATE/DELETE 없음) | OSS · 셀프호스트 · 클라우드, Apache-2.0. Free / $19 / $249 / Enterprise | [$24M Seed+A, 2025-10-28](https://mem0.ai/series-a) (Kindred, Basis Set, YC S24) | 66.0k |
| [Supermemory](https://github.com/supermemoryai/supermemory) | LLM 사실 추출 + 모순 해결 + 자동 망각, 정적·동적 사용자 프로필, 외부 커넥터 | 클라우드 또는 오프라인 단일 바이너리, MIT | $2.6M seed, 2025-10 | 30.9k |
| [Memori](https://github.com/MemoriLabs/Memori) | 기존 SQL DB 위에 사실·선호·규칙·요약으로 분류 | SDK · 클라우드(2026-03) · VPC · 온프렘, Apache-2.0 | 미확인 | 16.9k |
| [MemOS](https://github.com/MemTensor/MemOS) | 그래프 구조의 "MemCube", 하이브리드 검색 | 클라우드 · 셀프호스트(Neo4j+Qdrant) · 로컬 SQLite, Apache-2.0 | 미확인 | 11.6k |
| [Hindsight](https://benchmarks.hindsight.vectorize.io/) (Vectorize) | 기억 네 종류, retain(LLM 추출) / recall(4전략 RRF 융합) / reflect. Postgres+pgvector | 셀프호스트 · 임베디드 · 클라우드, MIT | 미확인 | 30.6k |

### 엔터프라이즈 · 그래프

| 제품 | 구조 | 비고 |
|---|---|---|
| [Zep / Graphiti](https://github.com/getzep/graphiti) | **이중 시간축 지식 그래프**: 사실마다 유효 기간. 쓸 때 LLM 이 에피소드에서 엔티티·사실 추출. 의미·BM25·그래프 순회 혼합 | Graphiti 31.2k★ Apache-2.0. Zep 은 SaaS · BYOC, "context graph" 로 포지셔닝. 투자 금액 미확인 |
| [Cognee](https://github.com/topoteretes/cognee) | 그래프 + 벡터, LLM 추출 파이프라인. v1.0(2026-06)은 Postgres 하나로 동작 | 31.0k★ Apache-2.0, [$7.5M seed 2026-02](https://www.cognee.ai/blog) |
| Engram (Chris Ré 등) | 검색이 아니라 **회사 문맥으로 모델을 미리 학습**시키는 "learned memory" | $98M, 2026-06 (기사 기준, 1차 출처 미확인) |
| [Letta](https://github.com/letta-ai/letta) (구 MemGPT) | 에이전트가 도구로 자기 메모리 블록을 고쳐 씀. Letta Code 는 git 기반 MemFS + sleep-time 반성 실행 | 24.9k★ Apache-2.0, $10M seed 2024-09 (이후 라운드 미확인) |
| [LangMem](https://github.com/langchain-ai/langmem) | 대화 중 도구 호출 또는 백그라운드 추출·통합, LangGraph BaseStore | 1.7k★ MIT |

### 클라우드 플랫폼 관리형

- [Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html): 단기 이벤트와 장기 기록. 추출 전략(내장 / 오버라이드 / 자체 관리)으로 자동 추출하고 의미 검색. **2025-10-13 GA.**
- [Vertex AI Memory Bank](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview): LLM 추출·통합, 개정 추적. preview.
- [Microsoft Foundry memory](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/what-is-memory): 사용자 프로필 · 대화 요약 · 절차 기억, 충돌 해결. preview.
- 셋 다 겨냥하는 건 코딩이 아니라 **챗 에이전트 개인화**입니다. 예외는 Anthropic 의 [Managed Agents memory stores](https://platform.claude.com/docs/en/managed-agents/memory) 로, 워크스페이스 단위 문서를 샌드박스에 디렉터리로 붙이고 변경마다 불변 버전을 남깁니다.

### 코딩 에이전트

하네스 내장:

| 하네스 | 저장 | 누가 쓰나 | 불러오기 |
|---|---|---|---|
| [Claude Code](https://code.claude.com/docs/en/memory) | CLAUDE.md 계층 + `~/.claude/projects/<slug>/memory/` (MEMORY.md 색인 + 주제 파일) | CLAUDE.md 는 사용자, auto memory 는 에이전트 | CLAUDE.md 전체, MEMORY.md 는 앞 200줄/25KB. **검색 인덱스 없음** |
| [Codex](https://learn.chatgpt.com/docs/customization/memories?surface=app) | `~/.codex/memories/` 생성 파일 | 과거 대화에서 백그라운드 자동 추출(비밀 제거) | 새 세션에 주입. **기본 꺼짐**, "필수 규칙은 AGENTS.md 에" |
| [GitHub Copilot Memory](https://docs.github.com/en/copilot/concepts/agents/copilot-memory) | 서버 측, **인용 근거 포함** | 에이전트 추론, 쓰기 권한 사용자의 행동에서만 | 쓰기 전에 **현재 브랜치와 대조 검증**, **28일 미사용 시 자동 삭제**. public preview |
| [Gemini CLI](https://geminicli.com/docs/tools/memory/) | GEMINI.md 계층 + 프로젝트별 비공개 폴더 | `save_memory` 또는 요청 시 파일 편집. 실험적 Auto Memory | 계층 전체 로드 |
| [Kiro](https://kiro.dev/docs/steering/) | `.kiro/steering/` 마크다운 | 사용자 | always / glob / manual / auto. **자동 메모리 없음** |
| [Devin Desktop (구 Windsurf)](https://docs.devin.ai/desktop/cascade/memories) | `~/.codeium/windsurf/memories/` | 자동 생성 | 메모리는 **"legacy"** 로 분류, 새 에이전트는 미지원 |
| [Cursor](https://cursor.com/docs/rules) | `.cursor/rules` + AGENTS.md | 사용자 | Memories(1.0 베타, 2025-06)는 **현재 문서에서 사라짐**. 제거 공지는 못 찾음 |

OSS · MCP 서버:

| 프로젝트 | 저장 · 검색 | ★ | 비고 |
|---|---|---|---|
| [claude-mem](https://github.com/thedotmack/claude-mem) | 훅으로 도구 사용 전부 캡처 → LLM 압축. SQLite FTS5 + Chroma 하이브리드, MCP 로 검색 → 타임라인 → id 조회 | 94.7k | Apache-2.0, 관찰기는 30일 체험 후 구독. 한국어 모드는 **출력 언어**만 |
| [MemPalace](https://github.com/MemPalace/mempalace) | 원문 그대로 ChromaDB, MCP 도구 45개 | 59.3k | 2026-04 등장. 벤치마크 논란(§3) |
| [OpenViking](https://github.com/volcengine/OpenViking) (ByteDance) | 메모리를 파일·디렉터리로 두는 문맥 DB | 38.7k | AGPL-3.0 |
| [Serena](https://github.com/oraios/serena) | `.serena/memories/` 마크다운, list/read 만 | 29.8k | 의미 검색 없음 |
| [agentmemory](https://github.com/rohitg00/agentmemory) | 훅 캡처, SQLite, BM25 + 벡터 + 그래프 RRF | 28.9k | CJK 분절기는 중국어·일본어용만 |
| [beads](https://github.com/gastownhall/beads) | 작업·이슈 기억, Dolt + JSONL | 27.4k | |
| TencentDB Agent Memory | 팀 허브(Wiki + CodeGraph), ACL | 27.3k | |
| [EverOS](https://github.com/EverMind-AI/EverOS) | 마크다운 + SQLite + LanceDB, 로컬 우선 | 13.2k | |
| [ByteRover CLI](https://github.com/campfirein/byterover-cli) (구 Cipher) | 마크다운 "context tree", 팀 공유는 클라우드 | 5.0k | Elastic-2.0 |
| [Basic Memory](https://github.com/basicmachines-co/basic-memory) | **마크다운이 원본**, SQLite/Postgres 인덱스 + FastEmbed + 리랭크 | 4.0k | AGPL, 클라우드 $15/월 |
| [memsearch](https://github.com/zilliztech/memsearch) (Zilliz) | 마크다운이 진실, Milvus 는 "다시 만들 수 있는 그림자 인덱스" | 2.7k | bge-m3 |
| [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) | sqlite-vec 등, 로컬 ONNX 임베딩, 자율 통합 | 2.0k | CJK 엔티티는 해시 id 로 대체 |
| [MCP reference `memory`](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) | 엔티티·관계·관찰 KG 를 JSONL 하나에 | (저장소 90.6k) | `search_nodes` 는 부분 문자열 일치 |
| [Graphiti MCP](https://github.com/getzep/graphiti) | 시간축 KG, 쓸 때 LLM 추출 | 31.2k | "experimental" 표기, LLM 키 필요 |

⚠️ Mem0 의 로컬 MCP 서버 **OpenMemory 는 2026-07-29 모노레포에서 제거됐습니다**(커밋 #6530).
이름은 세션 이식용 CLI 인 새 저장소 `mem0ai/openmemory` 로 넘어갔습니다.

## 2. 흐름

1. **코딩 에이전트는 사람이 고칠 수 있는 마크다운 파일로 수렴했습니다.** 주요 하네스가 전부 세션 시작에 파일을 싣는 방식이고,
   AGENTS.md 는 Claude Code · Codex · Cursor · Kiro · Devin 이 같이 읽는 공용 포맷이 됐습니다.
2. **자동 추출 메모리는 벤더 스스로 보조로 내리고 있습니다.** Windsurf 는 "legacy", Cursor 는 문서에서 사라졌고,
   Codex 는 규칙을 AGENTS.md 에 두라고 합니다. 남겨 둔 쪽(Copilot)은 인용 · 브랜치 대조 · 28일 만료로 신뢰 장치를 붙였습니다.
3. **"파일이 원본, 인덱스는 파생"** 구조가 늘고 있습니다(Basic Memory · memsearch · EverOS). 이 저장소의 hot/cold 와 같은 발상입니다.
   전부 캡처하는 쪽(claude-mem · agentmemory · MemPalace)은 SQLite/Chroma 에 BM25 + 벡터 하이브리드입니다.
4. **코딩 쪽에서는 순수 그래프 저장소의 힘이 빠지고 있습니다.** OpenMemory 제거가 한 신호이고, 연구 쪽 반례도 있습니다(§4).
5. **클라우드 플랫폼은 반대 방향**입니다. 관리형 LLM 추출 + 벡터 검색으로 챗 개인화를 겨냥합니다.
6. **2026년 새 프로젝트는 대부분 코딩 에이전트용**이고, MCP 서버에 라이프사이클 훅을 붙인 형태입니다.

## 3. 벤치마크

### 무엇을 재나

| 벤치마크 | 재는 것 | 알려진 한계 |
|---|---|---|
| [LoCoMo](https://arxiv.org/abs/2402.17753) (2024) | 약 300턴 · 최대 35세션 대화의 단일·다중 홉, 시간, 개방 도메인 QA | 대화가 16–26K 토큰이라 **문맥에 다 들어갑니다**. 정답 누락 · 화자 오귀속 · 지식 갱신 문항 없음. 전체 문맥 기준선(≈73%)이 Mem0(≈68%)를 이겼습니다 ([Zep](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)) |
| [LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025) | 500문항: 정보 추출 · 다중 세션 · 시간 추론 · **지식 갱신** · **답없음(abstention)** | 사실상 표준이 됐습니다 |
| DMR (MemGPT) | 깊은 기억 회수 | 포화(Zep 94.8 vs MemGPT 93.4) |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | 정확 회수 · 테스트 시 학습 · 장거리 이해 · **선택적 망각/충돌 해결** | 충돌 해결은 **전 시스템 ≤28%**(다중 홉). 정확 회수에서 **BM25(60.5%)가 임베딩 RAG(33.9–54.7%)를 이김** |
| [BEAM](https://arxiv.org/abs/2510.27246) | 최대 1,000만 토큰 대화 | 1M 문맥 모델도 대화가 길어지면 떨어집니다 |
| [TWIST](https://arxiv.org/abs/2609.28575) (2026-09) | 믿음이 바뀌는 지점에서의 행동 | 평평한 RAG 는 모순 76–97% 를 잡지만 안전한 내용 16–43% 를 잘못 표시. "recall 점수로는 안 보이는 교환" |
| [DolphinBench](https://arxiv.org/abs/2609.24971) (2026-09) | 약 50만 토큰 이력에서 작업 완수, **비용·지연 병기 필수** | 저자가 Mem0 창업자 — 벤더 벤치마크 |
| [VibeMemBench](https://arxiv.org/abs/2609.23570) (2026-09) | 코딩 에이전트: 90개 저장소 111과제 + 과거 궤적 3,634개 | 맞는 경험을 넣어 주면 +1.1~4.5pp 지만, 시스템이 스스로 만들고 찾게 하면 **12조합 중 11개가 메모리 끔을 못 이김** |

### 수치를 비교할 수 없는 이유

- LongMemEval-S 주장은 **92–97%**, LoCoMo 는 **90–94%** 로 포화됐습니다. 그런데 백본 모델 · 채점 모델 · S/M 변형 · 지표가 제각각입니다.
  - Supermemory 의 "97%" 는 Recall@20 인데, 비교 대상 Zep 71.2% 는 QA 정확도입니다.
  - Mem0 의 2026 수치(LoCoMo 92.5 · LongMemEval 94.4)는 README 스스로 "OSS 에 없는 관리형 플랫폼 최적화 포함"이라고 적습니다.
  - Mastra 는 GPT-5-mini 로 94.87%, 공식 모델 GPT-4o 로는 84.23% 입니다.
- **공개 분쟁**
  - Mem0 논문이 Zep 을 65.99% 로 보고 → Zep 은 두 화자를 한 사용자로 합치고 타임스탬프를 본문에 붙이고 검색을 직렬로 돌린 설정 오류라며 재측정 75.14% ±0.17.
  - [Letta](https://www.letta.com/blog/benchmarking-ai-agent-memory) (2025-08): grep · open 같은 **파일시스템 도구만 쓴 에이전트가 LoCoMo 74.0%** 로 Mem0(68.5%)를 이김. "메모리는 검색 장치가 아니라 에이전트가 문맥을 어떻게 다루느냐의 문제."
  - MemPalace: "snake oil" 논란 보도 후 README 에 마지막 0.6% 는 "시험에 맞춘 학습"이었다고 인정. 주장 수치 R@5 96.6 은 QA 가 아니라 회수 지표.
- **방법론 논문**
  - ["Same Ranking, Different Winner"](https://arxiv.org/abs/2605.24060): 정답 단위(턴 · 사실 · 요약)만 바꿔도 질의 83–94% 에서 nDCG 가 달라지고 **Mem0 와 MemoryOS 순위가 뒤집힘**.
  - [RENDER](https://arxiv.org/abs/2608.23568): 읽는 모델에 보여 주는 형식만 바꿔도 점수가 24.6–72.6점 움직임.

## 4. 연구 흐름

- **원문을 단순하게 검색하는 쪽이 계속 이깁니다.** [ReFind](https://arxiv.org/abs/2608.12888) 는 가공하지 않은 대화 로그에 렉시컬 검색만으로
  LongMemEval 93.2(S) / 89.3(M) 을 냈습니다. Letta 의 파일시스템 결과, Anthropic 의 파일 기반 [memory tool + context editing](https://claude.com/blog/context-management)
  (내부 평가 +39%, 토큰 84% 감소), MemoryAgentBench 의 BM25 결과가 같은 방향입니다.
- **그래프는 추출이 문제입니다.** ["Selective Forgetting"](https://arxiv.org/abs/2608.28978) 에서 LLM 엔티티 추출 그래프가
  평평한 벡터 기준선보다 낮았습니다(F1 0.417 vs 0.468). 턴을 엔티티로 쪼개며 원문 표현을 잃은 탓이고, 저자들은 그래프 일반이 아니라
  자기 추출 파이프라인의 문제라고 선을 긋습니다. 반대편 대표는 [A-MEM](https://arxiv.org/abs/2502.12110)(Zettelkasten 식으로 노트가 서로 링크하고 갱신).
- **오프라인 통합.** [Sleep-time compute](https://arxiv.org/abs/2504.13171) 는 테스트 시 연산 약 5배 절감에 정확도 +13–18%, 질의가 예측 가능할수록 효과가 큽니다.
- **긴 문맥 vs 검색.** 이력이 짧으면 긴 문맥이 이기지만, 약 10만 토큰을 넘거나 비용·지연이 중요하면 검색·압축이 필요하다는 게 현재 합의입니다
  (Chroma [Context Rot](https://www.trychroma.com/research/context-rot): 18개 모델 모두 입력이 길수록 정확도 하락).
- **메모리 포이즈닝이 주요 보안 주제가 됐습니다.**
  - [PMPA](https://arxiv.org/abs/2609.13889): **Claude Code 영속 메모리에 주입 성공률 66.9%**, 세션 간 81.7%. 한번 오염되면 프롬프트 수준 방어는 거의 효과 없음.
  - ["Price of Safety"](https://arxiv.org/abs/2609.22818): 쓰기 시점 방어(정제 · 출처 검증)는 정상 동작 비용이 측정되지 않음. 리랭커 기반 읽기 시점 방어는 정확도 −4.4점, 정상 기억 33.6% 오격리.
  - 출처(provenance) 기록이 다음 단계로 거론됩니다.

## 5. 한국어 · CJK

- **한국어나 CJK 트랙이 있는 공개 메모리 벤치마크는 없습니다.**
- 제품의 "다국어 지원"은 다국어 임베딩이나 출력 언어 수준입니다. 클라우드 플랫폼 셋은 언어별 검색을 문서화하지 않았습니다.
  **한국어 형태소 분석 · 띄어쓰기 · 복합명사를 다루는 렉시컬 검색은 거의 없습니다.**
- 예외이자 가장 가까운 비교 대상: [**PlateerLab/synaptic-memory**](https://github.com/PlateerLab/synaptic-memory) — 한국 회사 프로젝트, KG + MCP,
  **Kiwi 형태소 FTS** + BM25 + usearch 벡터, 색인 시 LLM 호출 없음, 한국어 FTS 벤치마크 포함. 33★, 마지막 push 2026-07-03.
- 외부 한국어 평가셋 후보: NAVER [**CareCall-Memory**](https://github.com/naver-ai/carecall-memory)
  ([논문](https://arxiv.org/abs/2210.08750), EMNLP Findings 2022). 한국어 다회차 대화 3,581건, **낡은·중복 기억의 갱신·제거**에 초점이라
  supersede 문제와 겹칩니다. 한국어 원본은 신청서가 필요하고 라이선스는 확인하지 않았습니다.

## 6. 이 저장소의 위치

**근거가 받쳐 주는 선택**

- hot 마크다운을 원본으로 두고 렉시컬 레인을 강하게 가져가는 설계 — §2-1·3, §4 첫 항목과 같은 방향입니다.
- 개념 그래프는 LLM 자동 추출이 아니라 **원문 위에 에이전트가 링크를 얹는** 방식이라 §4 의 그래프 반례가 그대로 적용되지 않습니다(A-MEM 쪽).
- `supersedes` 는 업계가 가장 약한 지점(충돌 해결 ≤28%)을 겨냥합니다.
- 비밀 거부와 "기록이지 지시가 아님" 표시는 비용이 거의 없다고 측정된 쓰기 시점 방어에 해당합니다.
- 게이트 비용 곡선처럼 정확도와 지연을 같이 재는 방식은 DolphinBench 등이 요구하는 보고 형태와 같습니다.

**약점**

- 채택 규모는 비교가 안 됩니다(claude-mem 95k★ · MemPalace 59k★).
- 수치가 전부 자체 known-item 픽스처의 R@k 라 공개 벤치마크와 비교할 수 없습니다. 게다가 §3 방법론 논문들은 R@k 가 답변 품질로 1:1 로
  이어지지 않고, 정답 단위만 바꿔도 순위가 뒤집힌다고 봅니다 — `measurements.md` 가 측정마다 조건을 적는 이유와 같은 문제입니다.

**후보 (합의된 계획 아님)**

1. "현재판이 supersede 된 옛판보다 위에 오는가"를 따로 재는 지표 — R@k 로는 이 축이 보이지 않습니다(TWIST 식 오표시율도 함께).
2. CareCall-Memory 를 외부 한국어 평가셋으로 쓸 수 있는지 라이선스 확인.
3. 쓰기 출처(provenance) 기록.
4. synaptic-memory 와 한국어 FTS 설계 비교.
