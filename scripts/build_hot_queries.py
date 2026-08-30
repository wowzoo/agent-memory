"""Build the hot-tier paraphrase fixture from a hand-authored case list.

Why a builder instead of a hand-written JSON: a hot case keys on `path` + `heading`,
and both are long strings copied out of the store. Transcribing them by hand is how
a fixture ends up with cases that silently stop resolving. Here the case list carries
only (project, file, topic prefix) and the exact keys are read back from the DB, so a
typo fails loudly at build time instead of showing up as a skipped case later.

    python scripts/build_hot_queries.py            # writes tests/data/hot_queries.json
    python scripts/build_hot_queries.py --dry-run  # resolve only, write nothing

Authoring rules these cases follow (from project-agent-memory-build):
  - `TOPIC_WEIGHT` is 3.0, so a query lifted from the section heading is the most
    optimistic query there is. Queries here are phrased as vague recollection plus
    the identifiers a user would actually retain (version, commit, product, file).
  - Each case is anchored on what separates the target from its near-neighbours, so
    a miss means retrieval failed rather than that the query was unanswerable.
  - Facts held by two projects are excluded: the gold would be arbitrary. That rules
    out every file name appearing under more than one project, and the Hangul-
    corruption subject (agent-memory and aidlc-v2-patch both hold it).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import db  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "hot_queries.json"

# (block, project, file, topic prefix, query, authored_from)
#
# block  hot_only      the fact lives only in a hot section
#        hot_over_cold both tiers hold it; gold is the hot copy, which is what the
#                      recall contract promises to rank first
# authored_from
#        id_recall     identifiers + recall phrasing, no heading wording reused
#        topic_echo    the query unavoidably reuses distinctive heading words, so
#                      this case gets the 3x topic boost and reads optimistically
CASES = [
    # ---------------------------------------------------------------- agent-memory
    ("hot_only", "agent-memory", "project-agent-memory-build.md",
     "rerank depth 20",
     "리랭킹 후보를 20보다 깊게 볼지 결론이 어떻게 났지", "id_recall"),
    ("hot_only", "agent-memory", "project-agent-memory-build.md",
     "프로젝트 기본 범위 — 판정 완료",
     "회수 기본값으로 지금 워크스페이스만 뒤지게 할지 정했었나", "id_recall"),
    ("hot_only", "agent-memory", "project-agent-memory-build.md",
     "리랭커 채택 완료",
     "크로스 인코더를 항상 켜기로 했나 옵션으로 뒀나", "id_recall"),
    ("hot_only", "agent-memory", "project-agent-memory-build.md",
     "다중 프로젝트 hot 삭제 결함",
     "다른 워크스페이스 메모리 파일을 지웠는데 색인이 남던 버그", "id_recall"),
    ("hot_only", "agent-memory", "project-agent-memory-build.md",
     "cold 본문에 entities 배열이 섞여 저장된다",
     "저장된 본문 안에 엔티티 배열이 같이 들어가 있던 건 어떻게 처리했지", "topic_echo"),

    # ---------------------------------------------------------- old-new-compare-v2
    ("hot_only", "old-new-compare-v2", "kiro-credit-usage-measurement.md",
     "✅ Kiro CLI",
     "Kiro CLI 사용량이 어느 sqlite 에 쌓이나", "id_recall"),
    ("hot_only", "old-new-compare-v2", "kiro-credit-usage-measurement.md",
     "❌ Kiro IDE",
     "Kiro IDE 는 크레딧이 파일로 안 남는다고 확인한 근거", "id_recall"),
    ("hot_only", "old-new-compare-v2", "cursor-harness-has-installer.md",
     "cursor-harness-has-installer",
     "하네스 일곱 중 하나만 인스톨러를 갖는다던 게 어느 하네스지", "id_recall"),
    ("hot_only", "old-new-compare-v2", "version-channel-conflict.md",
     "version-channel-conflict",
     "어댑터가 stdin 을 먼저 읽게 바뀐 상류 이슈 닫혔나", "id_recall"),
    ("hot_only", "old-new-compare-v2", "issue-495-superseded-by-658.md",
     "issue-495-superseded-by-658",
     "내 PR 이 머지 안 되고 상류 다른 PR 로 대체됐던 건", "id_recall"),
    ("hot_only", "old-new-compare-v2", "upstream-roadmap-is-theme-based.md",
     "upstream-roadmap-is-theme-based",
     "상류 로드맵에서 마이너 버전별 약속이 없어진 개편", "id_recall"),

    # ------------------------------------------------------ aidlc-scale-assessment
    ("hot_only", "aidlc-scale-assessment", "skill-state-and-facts.md",
     "폰트는 CDN 에서 온다",
     "산출물 폰트가 이 기계에 없어서 네트워크로 받는 문제 어떻게 정리했지", "topic_echo"),
    ("hot_only", "aidlc-scale-assessment", "skill-state-and-facts.md",
     "it.6 에서 확정된 것",
     "표면이 몇 배로 늘었을 때 전수 대조해서 거짓 양성이 몇 건이었나", "topic_echo"),
    ("hot_only", "aidlc-scale-assessment", "skill-eval-method.md",
     "합성 집합은 스킬을 가리지 못한다",
     "결함을 일부러 심어 만든 입력으로 재면 안 되는 이유", "id_recall"),
    ("hot_only", "aidlc-scale-assessment", "skill-eval-method.md",
     "비용은 서브에이전트로만 측정된다",
     "메인 루프로 돌리면 토큰 사용량이 안 남는다는 것", "topic_echo"),

    # ----------------------------------------------- ai-dlc-1-day-workshop-working
    ("hot_only", "ai-dlc-1-day-workshop-working", "aidlc-bolt-unit-model.md",
     "🔴 모델이 두 개다",
     "Bolt 단위가 상류 용어집과 엔진 동작에서 어긋난다던 것", "id_recall"),
    ("hot_only", "ai-dlc-1-day-workshop-working", "aidlc-v2-claude-install-prompt.md",
     "핵심 함정",
     "cp -r 로 설치할 때 디렉터리가 한 겹 더 생기던 함정", "id_recall"),
    ("hot_only", "ai-dlc-1-day-workshop-working", "workshop-v2-resume.md",
     "check-stale.js DENY 토큰",
     "검증에서 금지어로 걸리는 옛 agent 이름들", "id_recall"),
    ("hot_only", "ai-dlc-1-day-workshop-working", "workshop-v2-history.md",
     "2026-07-26~28",
     "센서 개수 판정이 뒤집힌 회차", "topic_echo"),

    # ------------------------------------------------------------- aidlc-v2-docs
    ("hot_only", "aidlc-v2-docs", "feedback_deck_no_per_component_reveal.md",
     "feedback_deck_no_per_component_reveal",
     "장 안에서 카드를 클릭마다 하나씩 띄우지 말라는 규칙", "id_recall"),
    ("hot_only", "aidlc-v2-docs", "project_aidlc_v2_docs_repo.md",
     "🔴 이 repo 의 제1규칙",
     "pptx 를 직접 고치면 안 되는 이유", "id_recall"),
    ("hot_only", "aidlc-v2-docs", "feedback_no_italic_use_regular.md",
     "feedback_no_italic_use_regular",
     "이탤릭 쓰지 말라고 했던 지시", "id_recall"),

    # ------------------------------------------------------------ aidlc-v2-patch
    ("hot_only", "aidlc-v2-patch", "reference_kiro_unified_limits.md",
     "A. 안 된다",
     "위임 agent 에서는 훅이 아예 안 발화한다던 여덟 칸 전수 조사", "id_recall"),
    # Heading moved 2026-08-21: that project's own session folded the finding into
    # `🔴 열린 항목` item 1. Re-anchored only after checking the new section still
    # carries the fact (same-machine `/status` twice, us-west-2 -> us-east-1,
    # opus-5 -> opus-4-8); matching a heading without that check labels the wrong
    # section as gold, which is worse than a case that fails to resolve.
    ("hot_only", "aidlc-v2-patch", "project_v2_claude_harness_tree.md",
     "🔴 열린 항목",
     "배포본 설정 파일이 내 모델·리전 설정을 덮어쓰던 발견", "id_recall"),
    ("hot_only", "aidlc-v2-patch", "reference_aidlc_plugin_mechanism.md",
     "⭐첫 자작 plugin 실증",
     "직접 만든 첫 플러그인이 뭐였지", "id_recall"),
    ("hot_only", "aidlc-v2-patch", "project_aidlc_dashboard.md",
     "⭐ stage 별 산출물 아코디언",
     "stage 이름 누르면 산출물 목록 펼치는 기능 넣은 작업", "topic_echo"),

    # ------------------------------------------ hot_over_cold (gold = the hot copy)
    # The cold chronicle still says a separate blank-page checker was created; the
    # hot section is the correction (it was folded into deck_to_pdf.py). If cold
    # wins here the reader gets the superseded answer.
    ("hot_over_cold", "aidlc-v2-docs", "project_html_deck_print_overflow.md",
     "✅ 이제는",
     "빈 장 검사를 따로 돌리는 스크립트로 뒀나 합쳤나", "id_recall"),
    ("hot_over_cold", "aidlc-v2-docs", "project_html_deck_channel.md",
     "2026-08-21 에 한 일",
     "레일 접는 토글 넣은 날 무엇을 같이 했지", "id_recall"),
    ("hot_over_cold", "aidlc-v2-docs", "project_aidlc_v2_docs_repo.md",
     "무엇이 왜 갈라졌나",
     "발표덱을 왜 별도 repo 로 뗐지", "id_recall"),
    ("hot_over_cold", "agent-memory", "project-recall-query-collection.md",
     "첫 적용",
     "드롭 규칙 처음 적용했을 때 후보 몇 건 중 몇 건 남겼지", "id_recall"),

    # These eight were found mechanically rather than by eye: a cold archive entry and
    # exactly one hot section sharing two or more identifiers (`#\d+`, `x.y.z`, a short
    # commit hash) inside one project. Sharing two rules out the coincidence that
    # sharing one does not -- `AUDIT` alone matched a MEMORY.md index.
    # Each query is anchored on an identifier that is in the hot body but NOT in the
    # hot heading, so neither side collects the 3x topic boost and the two tiers
    # contest the case on body text alone.
    ("hot_over_cold", "agent-memory", "project-agent-memory-build.md",
     "2026-08-09 검증 완료분",
     "MCP 도구 다섯 개를 실제 세션에서 호출해 검증한 회차", "id_recall"),
    ("hot_over_cold", "aidlc-v2-patch", "project_aidlc_workshop_content.md",
     "🔴 bash rc 안내가 틀렸었다",
     "비대화형 셸이 어느 rc 를 읽는지 안내가 틀렸던 것", "id_recall"),
    ("hot_over_cold", "aidlc-v2-patch", "project_v2_to_kiro_ide.md",
     "이 세션 후반",
     "MARM 을 2.36.0 으로 올려 개념 그래프 설치한 작업", "id_recall"),
    ("hot_over_cold", "aidlc-v2-patch", "reference_upstream_kiro_ide_hook_wiring_gap.md",
     "영향 — 두 세대가 IDE",
     "상류 2.5.40 훅 변경으로 우리 세대가 IDE 에서 안 먹던 것", "id_recall"),
    ("hot_over_cold", "aidlc-v2-patch", "reference_claude_harness_run_cost.md",
     "🔴🔴 2026-08-05 대정정",
     "transcript 수치가 1.7배 부풀어 있던 대정정", "topic_echo"),
    ("hot_over_cold", "old-new-compare-v2", "aidlc-test-baseline-runstate.md",
     "aidlc-test-baseline-runstate",
     "#417 과 #455 판정이 정리된 곳", "id_recall"),
    ("hot_over_cold", "old-new-compare-v2", "release-cadence-report-plan.md",
     "release-cadence-report-plan",
     "2.2.14 에서 2.2.18 까지 릴리스 간격 정리한 계획", "id_recall"),
    ("hot_over_cold", "old-new-compare-v2", "kiro-credit-usage-measurement.md",
     "❌ 공식 프로그래매틱 접근",
     "kiro-cli 2.18.0 에서 usage 서브커맨드가 먹히나", "id_recall"),
]

COMMENT = [
    "Hot-tier paraphrase queries. Kept apart from paraphrase_queries.json because",
    "that file's 72 cases are the denominator under every recall A/B so far, and",
    "apart from collected_queries.json because those are field drops rather than",
    "authored cases.",
    "Reason this file exists: the 72 cases are 100% cold gold while the store is",
    "hot-heavy, so neither the two-tier number nor MIN_TIER_SLOTS can be judged on",
    "them -- raising a cold slot raises the score by definition. These cases put",
    "gold on the hot side.",
    "Blocks: hot_only (fact lives only in hot) and hot_over_cold (both tiers hold",
    "it, gold is the hot copy, which is the precedence the recall contract",
    "promises). Cases carry `project` so a per-project breakdown is possible.",
    "Bias: absolute numbers here read optimistically for cases tagged topic_echo,",
    "because TOPIC_WEIGHT is 3.0 and those queries reuse heading words. Read the",
    "id_recall subset as the honest figure and the split as the bias estimate.",
    "Built by scripts/build_hot_queries.py -- edit the case list there, not here.",
]


def tilde(path: str) -> str:
    """The stored absolute path with `$HOME` folded back to `~`, or unchanged."""
    home = str(Path.home())
    return f"~{path[len(home):]}" if path.startswith(home + "/") else path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = db.connect()
    cases, failed = [], []
    for block, project, fname, prefix, query, authored in CASES:
        rows = con.execute(
            "SELECT path, topic FROM memories WHERE tier = 'hot' AND project = ? "
            "AND path LIKE ? AND substr(topic, 1, ?) = ?",
            (project, f"%/{fname}", len(prefix), prefix),
        ).fetchall()
        if len(rows) != 1:
            failed.append((project, fname, prefix, len(rows)))
            continue
        cases.append({
            "query": query,
            # `~`-relative, because this file is committed: the store keeps absolute
            # paths, and writing them here puts one machine's home directory into the
            # repository. Readers expand it (`eval_paraphrase.resolve_hot`).
            "path": tilde(rows[0]["path"]),
            "heading": rows[0]["topic"],
            "block": block,
            "project": project,
            "authored_from": authored,
        })

    for project, fname, prefix, n in failed:
        print(f"  ! {n} matches (need 1): {project} {fname} :: {prefix}")
    print(f"resolved {len(cases)}/{len(CASES)} cases")
    for name, want in (("hot_only", "hot_only"), ("hot_over_cold", "hot_over_cold")):
        print(f"    {name:14} {sum(c['block'] == want for c in cases)}")
    print(f"    {'id_recall':14} {sum(c['authored_from'] == 'id_recall' for c in cases)}")
    print(f"    {'topic_echo':14} {sum(c['authored_from'] == 'topic_echo' for c in cases)}")

    if failed:
        print("\nnot writing: every case must resolve to exactly one section")
        sys.exit(1)
    if args.dry_run:
        print("\ndry run, nothing written")
        return

    OUT.write_text(json.dumps({"_comment": COMMENT, "cases": cases},
                              ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
