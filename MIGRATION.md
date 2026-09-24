# Migration note — agent-memory

이 디렉터리는 여러 MCP 클라이언트가 사용하는 `agent-memory` 서버를 새 workspaces 구조에서 운영하기 위해 만든 독립 복사본입니다.

## Locations

- 원본 소스: `/Users/wowzoo/Development/agent-memory` — ⚠️ 2026-09-13 확인 시 이미 없음(아래 `## Rollback`)
- 새 소스: `/Users/wowzoo/workspaces/platforms/agent-tools/memory/agent-memory`
- 공유 운영 데이터: `/Users/wowzoo/.agent-memory`

원본 소스와 공유 운영 데이터는 이 복사 과정에서 수정·삭제하지 않았습니다.

## Git baseline

- Branch: `main`
- HEAD at copy: `29d43b9a5bb6cac40a2101d42b538a980266850d`
- Origin: `git@github.com:wowzoo/agent-memory.git`
- Source worktree at copy: clean

## Excluded from the copy

다음 항목은 재생성 가능하거나 위치 종속적이므로 복사하지 않았습니다.

- `.venv/`
- `.pytest_cache/`
- `*.egg-info/`
- `__pycache__/`
- `*.pyc`

새 실행 환경은 이 디렉터리에서 `uv sync`로 생성합니다.

## Shared data boundary

서버의 기본 데이터 경로는 소스 저장소 밖의 `/Users/wowzoo/.agent-memory`입니다.

- SQLite DB: `/Users/wowzoo/.agent-memory/memory.db`
- Embedding model: `/Users/wowzoo/.agent-memory/models/multilingual-e5-small-ko`
- Reranker model: `/Users/wowzoo/.agent-memory/models/ko-reranker-int8`
- Recall logs: `/Users/wowzoo/.agent-memory/recall-log`

새 서버도 별도 환경 변수가 없으면 동일한 DB와 모델을 사용합니다. 운영 데이터는 소스 이관과 별도로 이동하지 않습니다.

## Consumer cutover

새 실행 파일 준비 후 다음 소비자의 command를 새 `.venv/bin/agent-memory`로 전환합니다.

- Claude Code 프로젝트 범위 등록 8곳 (`/Users/wowzoo/.claude.json`)
- Codex 전역 등록 1곳 (`/Users/wowzoo/.codex/config.toml`)
- Kiro/KiroCrew는 현재 활성 등록이 없어 변경하지 않음

실행 중인 기존 MCP 프로세스는 종료하지 않고, 새 세션부터 새 실행 파일을 사용하게 합니다.

## Rollback

⚠️ **아래 절차는 더 이상 유효하지 않습니다.** 원본 소스 `/Users/wowzoo/Development/agent-memory`는
2026-09-13 확인 시 이미 사라졌고(상위 `~/Development`째 없음), 그 `.venv/bin/agent-memory`로 command를
되돌리면 서버가 기동하지 않습니다.

> ~~전환 문제가 발생하면 Claude·Codex의 command를 `/Users/wowzoo/Development/agent-memory/.venv/bin/agent-memory`로
> 되돌립니다. 기존 소스와 가상환경을 보존하므로 즉시 롤백할 수 있습니다.~~

지금 되돌릴 곳은 다른 디렉터리가 아니라 이 저장소의 이전 커밋입니다. 이관 시점 HEAD는
`29d43b9`(위 `## Git baseline`)이고 origin에도 있습니다. command는 그대로 두고 이 디렉터리에서
해당 커밋을 체크아웃한 뒤 `uv sync`, 실행 중인 MCP 서버를 재시작합니다. 스키마는 전진만 하므로
(`db._migrate`) 새 버전이 한 번이라도 연 DB를 옛 코드로 여는 경우는 따로 확인해야 합니다.

## Cutover completed — 2026-09-07

- 새 `.venv` 생성: CPython 3.13.5, `uv sync --frozen --no-dev`
- 핵심 테스트: 27 passed
- Claude Code: 기존 프로젝트 8곳 + 새 프로젝트 1곳, 모두 새 command 사용
- Codex: 전역 등록 1곳, 새 command 사용
- Kiro/KiroCrew: 활성 등록 없음, 변경 없음
- 공유 라이브 DB read-only 확인: memories 1666, sessions 203, entities 2686, relations 3080
- 동시 MCP 클라이언트 스모크: 2개 클라이언트가 같은 scratch DB 복사본에서 lexical recall 성공
- 기존 MCP 프로세스와 기존 소스는 종료·삭제하지 않음
