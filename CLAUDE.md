# GraphLM — AI 협업 가이드

이 파일은 AI(Claude, Copilot 등)가 프로젝트를 이해하기 위한 **목차이자 진입점**입니다.
모든 정보를 여기에 담지 않고, 필요한 시점에 해당 문서를 참조하도록 설계했습니다.

## 프로젝트 한 줄 요약

그래프(Graph) 구조와 언어 모델(Language Model)을 결합한 연구·실험 프로젝트.
주 언어는 Python, 실험은 Jupyter Notebook(`.ipynb`)으로 진행한다.

## 빠른 참조 (목차)

작업 유형별로 필요한 문서만 읽으세요. **전부 읽지 마세요.**

| 작업 | 참조 문서 | 핵심 내용 |
|------|-----------|-----------|
| 코드 스타일 확인 | `.claude/rules/06-code-style.md` | PEP 8, ruff, 네이밍, 한국어 커밋 |
| 에러 처리 | `.claude/rules/04-error-handling.md` | 예외 타입, 재시도, 로깅 필드 |
| 테스트 작성 | `.claude/rules/05-testing.md` | `tests/` 디렉토리 구조, 커버리지 70% |
| 아키텍처 이해 | `.claude/rules/01-architecture.md` | 모듈 레이아웃, 데이터 흐름, 노트북 구성 |
| **AI 작업 진행** | `.claude/rules/07-workflow.md` | **자율 진행 / commit-per-TODO / PR 자동 / 권한 최소화** |
| CI/머지 게이트 | `docs/ci/conventions.md` | Required checks, CODEOWNERS, Ruleset |
| Status check 이름 | `docs/ci/status-checks.md` | 단일 소스, 변경 절차 |
| 논문/레퍼런스 요약 | `docs/papers/README.md` | 디렉토리 구조, 파일명/frontmatter 규칙, 추가 절차 |

## 반드시 지킬 규칙 (CI가 강제함)

이 규칙을 어기면 CI가 실패하여 머지가 차단됩니다. "하지 마"가 아니라 **못 합니다.**

1. **커밋 메시지**: `[FEAT]:` / `[FIX]:` / `[REFAC]:` / `[DOCS]:` / `[CHORE]:` 로 시작. 한국어.
2. **PR 타이틀**: `[카테고리#이슈번호] 제목` (CI 정규식 강제). 카테고리는 위 5종, 이슈번호 누락이나 카테고리 오타 시 머지 차단.
3. **포맷**: `ruff format .` 로 포맷 정리 후 커밋.
4. **임포트 가능 여부**: 패키지가 import 가능한 상태 유지 (`python -c "import graphlm"` 통과).
5. **테스트**: `pytest` 통과. 커버리지 70% 이상.
6. **린트**: `ruff check .` + `nbqa ruff notebooks/` + `scripts/check-papers-frontmatter.py` 통과.

## 빌드/테스트 명령어

```bash
make install-notebook  # uv 로 .venv + dev/notebook 의존성 설치
make kernel            # .venv 를 Jupyter 커널로 등록 ("GraphLM (uv .venv)")
make test              # pytest 실행 (uv run)
make coverage          # 커버리지 리포트
make lint              # ruff + nbqa (notebooks 포함) + docs/papers frontmatter
make fmt               # ruff format
make nb-clean          # 노트북 출력 셀 정리 (nbstripout)
make papers-lint       # docs/papers frontmatter 단독 검증
```

모든 명령은 내부적으로 `uv run` 으로 `.venv` 를 사용합니다.
직접 셸에서 실행하려면 `source .venv/bin/activate` 후 진행.

## 디렉토리 구조 (요약)

```
src/graphlm/    → Python 패키지 (모델·그래프·유틸 구현)
notebooks/      → Jupyter 실험 노트북 (.ipynb)
tests/          → pytest 테스트 (src/graphlm/ 미러링)
data/           → 실험 데이터 (gitignore — 큰 파일은 별도 관리)
docs/ci/        → CI 운영 규약, status check 단일 소스
docs/papers/    → 논문/레퍼런스 요약 (md, frontmatter 규칙 + 주제별 폴더)
.claude/rules/  → 상세 개발 규칙 (위 목차 참조)
```

## 작업 전 체크리스트

- [ ] 관련 규칙 문서를 **목차에서 찾아** 읽었는가? (전체 읽기 금지)
- [ ] 커밋 메시지가 `[카테고리]: 한국어 설명` 형식인가?
- [ ] `make fmt && make lint && make test` 를 로컬에서 통과했는가?
- [ ] 노트북은 `make nb-clean` 후 커밋하는가? (출력 셀 diff 폭주 방지)

## AI 작업 진행 규약

상세는 [`.claude/rules/07-workflow.md`](.claude/rules/07-workflow.md). 핵심 6 규약:

1. **이슈 먼저 생성** — 코드 수정 시작 전 GitHub 이슈 생성. 큰 작업은 메인 + sub-issue N개로 분할 후 모두 사전 생성 (Sub-issue Relation 활성화). PR 직전 ad-hoc 이슈 금지.
2. **자율 진행** — 시스템 변경 / destructive 권한 / 외부 영향 / 모호 영역만 사용자 확인. 그 외는 쿼리 의도 기반 자율 진행.
3. **Commit-per-TODO** — 별 언급 없으면 논리적 변경 단위마다 commit (메시지 컨벤션 준수).
4. **PR 자동 생성** — 작업 완료 직후 컨벤션 + 템플릿 준수해서 `Closes #<sub-issue>` 포함 PR 자동 생성. 마지막 sub-issue PR 에서 메인 이슈도 close.
5. **권한 사용 최소화** — 새 permission / 외부 도구 / 의존성은 작업 완수에 불가피한 경우에만.
6. **Label · Issue Type 부여 필수** — 이슈는 **issue prefix** 기준 Label + Type (`[FEATURE]→enhancement/Feature`, `[REFACTOR]→refactor/Task`, `[CHORE]→chore/Task`, `[DOCS]→documentation/Task`, `[FIX]→bug/Bug`, `[HOTFIX]→bug+hotfix/Bug`). PR Label 은 그 PR 이 닫는 이슈의 Label 과 동일. **부여 수단: `scripts/gh-meta.sh issue <N>` / `scripts/gh-meta.sh pr <N>` — 수동 `gh api graphql` 대신 항상 이 스크립트 사용**. 표기 체계 3분리 (commit `[FEAT]:` / PR `[FEAT#N]` / issue `[FEATURE]`) 는 [규약 6](.claude/rules/07-workflow.md) 참조.

## PR 생성 후 자동 동작

`gh pr create` 가 성공한 직후 사용자가 별도 지시하지 않아도 다음을 자동 수행한다:

1. **`@.claude/loop.md` 를 3분 주기 cron 으로 등록** — `CronCreate` 호출
   - cron 표현식: `*/3 * * * *`
   - prompt: `@.claude/loop.md 절차에 따라 PR #N 의 CI 와 코멘트를 점검하고 처리해줘.` (N = 방금 생성한 PR 번호)
   - recurring: `true`
2. 사용자에게 한 줄 보고 — cron job ID 와 본 PR url 포함

자동 등록 예외:
- 사용자가 명시적으로 "loop 등록하지 마" 라고 지시하면 생략
- draft PR 등 후속 polling 이 무의미한 케이스가 명백하면 사용자에게 묻고 진행

자동 종료는 `.claude/loop.md` 의 "자동 중단 (CI 완료 후 2회 연속 무동작 시)" 섹션이 처리한다.
