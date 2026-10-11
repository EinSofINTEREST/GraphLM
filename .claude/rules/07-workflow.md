# AI 작업 진행 규약 — 본 프로젝트 전용 사항

공통 규약은 사용자 레벨 하네스 **`~/.claude/rules/`** 에 있다 (2026-10-11 이관, 이슈 #84).
이 문서는 **공통 규약이 프로젝트에 위임한 값** 과 **본 저장소만의 추가 규칙** 만 적는다.
공통 규약과 충돌하면 이 문서가 우선한다. 단, 이 문서에는 **전용 값과 명시적 예외** 만 적는다 — 공통 규약 자체를 바꾸고 싶으면 이 문서에 덮어쓰지 말고 하네스 쪽을 고치고 결정으로 기록한다. 코드 자체의 규약은 [01-architecture.md](01-architecture.md) ~
[06-code-style.md](06-code-style.md).

> `~/.claude/rules/` 가 없는 환경 (CI 등) 에서는 [CLAUDE.md](../../CLAUDE.md) 의 "반드시 지킬 규칙" 과
> "AI 작업 진행 규약" 요약, 그리고 이 문서만 적용된다.

<br>

## 구 규약 번호 → 공통 하네스 대응표

기존 이슈 · PR · 문서 · 스크립트 주석 (`scripts/gh-meta.sh` 등) 이 "규약 N" 으로 참조한다. 번호는 그대로 쓴다.

| 구 규약 | 내용 | 공통 하네스 (`~/.claude/rules/`) | 본 프로젝트 전용 (아래 절) |
|---|---|---|---|
| 1 | 이슈 먼저 · sub-issue 분할 | `01-workflow.md` 1절, `03-conventions.md` Sub-issue 연결 | — |
| 2 | 자율 진행 · 예외 4영역 | `01-workflow.md` 2절 | 자율 영역 추가 항목 |
| 3 | Commit-per-TODO | `01-workflow.md` 3절 | 노트북 커밋 |
| 4 | PR 자동 생성 · 생성 후 피드백 | `01-workflow.md` 4절, `06-pr-feedback.md` | 도구 · `loop.md` 의 위치 |
| 5 | 권한 최소화 | `01-workflow.md` 5절 | Python 의존성 |
| 6 | Label · Issue Type · 접두사 3분리 | `03-conventions.md` | `scripts/gh-meta.sh` |
| — | 결정 기록 (ADR) | `07-decisions.md` | 저장 위치 (아래) |
| — | 오케스트레이션 · 역할 군 · 티어 | `02-orchestration.md`, 역할 정의 `~/.claude/agents/` | PR 작업 분배 표 |

<br>

## 규약 2 — 자율 진행 영역 추가 항목

공통 목록에 더해 다음도 확인 없이 한다:

- `scripts/` 하위 도구 실행, `make` · `scripts/lint.sh` (ruff + nbqa) 실행.
- 의존성 추가 (`pyproject.toml` 갱신, `uv`) — 단 신규 외부 패키지는 사전 확인 (규약 5).
- 노트북 출력 셀 정리 `make nb-clean`.

<br>

## 규약 3 — 노트북 커밋

- 노트북 (`.ipynb`) 은 **`make nb-clean` 후 커밋** 한다. 출력 셀이 diff 를 폭주시킨다.
- 로직은 `src/graphlm/`, 실험 흐름은 `notebooks/` — 한 커밋에 둘을 섞지 않는다 ([01-architecture.md](01-architecture.md)).

<br>

## 규약 4 · 6 — 도구

| 용도 | 수단 | 비고 |
|---|---|---|
| 이슈 Label + Issue Type | `scripts/gh-meta.sh issue <N>` | title prefix 로 자동 판정. 수동 `gh api graphql` 금지. Issue Type 은 `GRAPHLM_*_TYPE_ID` 환경변수가 설정된 경우에만 부여 — 조회 방법은 `scripts/gh-meta.sh` 상단 주석 |
| PR Label | `scripts/gh-meta.sh pr <N>` | 닫는 이슈의 Label 과 동기화 |
| 리뷰 코멘트 처리 완료 표시 | `scripts/pr-resolve-comments.sh <PR> <comment_id...>` | 👀 + resolve 일괄 |
| sub-issue 연결 | `gh api graphql` `addSubIssue` (공통 `03-conventions.md` 의 명령) | 스크립트 없음 — 직접 호출 허용 |
| Notion PR 동기화 | `.github/workflows/notion-pr-sync.yml` | 자동. 손으로 행을 만들지 않는다 |

### PR 생성 후 피드백 (하네스 기본값으로 전환, 이슈 #84)

- **세션이 살아 있으면** `Monitor` 로 CI · 코멘트를 감시하고 그 세션에서 처리한다 (공통 `06-pr-feedback.md`).
- **`gh pr create` 직후 `.claude/loop.md` 를 3분 cron 으로 자동 등록하던 구 동작은 폐지** 됐다. 새 세션이 diff 만 보고
  맥락을 잃고, 토큰이 누적되고, 자체 상태 기계가 필요했다. 사용자가 명시적으로 요청하지 않는 한 `CronCreate` 를 호출하지 않는다.
- [`.claude/loop.md`](../loop.md) 는 삭제하지 않고 **수동 보조 수단** 으로 남긴다 — 사용자가 `/loop` 로 요청할 때만 쓴다.
- [`.claude/pr-feedback.md`](../pr-feedback.md) (신규 PR 자동 리뷰 루프, `scripts/pr-feedback.sh`) 는 PR 생성과 무관한 별도 자동화다.
  사용자가 직접 등록할 때만 돈다.
- **세션을 닫은 뒤** 의 피드백은 사용자가 처리한다 — 본 저장소에는 `@claude` 멘션용 `claude.yml` 이 없다. 도입 여부는 별도 결정.

<br>

## 규약 5 — Python 의존성

- 신규 외부 패키지 (`pyproject.toml` · 시스템 패키지) 추가는 모호 영역 → 사용자 사전 확인. 명시 PR 로 분리한다 ([01-architecture.md](01-architecture.md)).
- 무거운 의존성 (torch 등) 은 이미 있는 것을 쓴다. 대체 라이브러리 도입은 결정 기록 대상.

<br>

## 결정 기록 위치

본 저장소에는 Notion ADR 데이터베이스가 없다. 공통 `07-decisions.md` 의 우선순위에 따라 **PR 본문 "결정" 섹션** 에 ADR 형식으로 적고,
핵심 패러다임 수준의 결정은 [CLAUDE.md](../../CLAUDE.md) "핵심 패러다임" 과 프로젝트 메모리에 반영한다.
전용 저장소 (`docs/decisions.md` 또는 Notion DB) 도입 여부는 별도 결정.

<br>

## 오케스트레이션 보충

- 역할 군 · 티어 · 모델 매핑 · 브리프 · 검수는 공통 `02-orchestration.md` 가 단일 출처. 역할 정의는 `~/.claude/agents/`.
- 서브 에이전트 브리프의 "제약" 에 넣는 포인터: 테스트 위치 · 마커 [05-testing.md](05-testing.md), NaN · 결정론 [04-error-handling.md](04-error-handling.md),
  스타일 [06-code-style.md](06-code-style.md), 코드/노트북 분리 [01-architecture.md](01-architecture.md). 실험 노트북을 서브 에이전트에 맡길 때는 커널 · 시드 고정 조건을 명시.
- 검수 재실행 명령: `make fmt` · `make lint` · `make test` (· `make nb-clean`).
- PR 템플릿에 "작업 분배" 섹션이 없으므로 "구현 내용" 아래에 표를 추가한다. 서브 에이전트를 쓰지 않았으면 "없음".

<br>

## 참고 자료

- 관련 규약: [06-code-style.md](06-code-style.md) — commit/PR 메시지 컨벤션 / [05-testing.md](05-testing.md) — 작업 단위 테스트 기준
- 관련 문서: [.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md), [docs/ci/conventions.md](../../docs/ci/conventions.md)
- 이슈 #84 — 공통 규약을 `~/.claude/rules/` 로 이관, cron 자동 등록 폐지
