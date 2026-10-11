# GraphLM — AI 협업 가이드

이 파일은 AI(Claude, Copilot 등)가 프로젝트를 이해하기 위한 **목차이자 진입점**입니다.
모든 정보를 여기에 담지 않고, 필요한 시점에 해당 문서를 참조하도록 설계했습니다.

## 프로젝트 한 줄 요약

**학습 중 Transformer 의 파라미터 수를 동적으로 조정** 하기 위해 모델 구조를 graph 로 다루는 연구·실험 프로젝트.
graph 표현은 도구일 뿐, 진짜 목적은 *training-time dynamic parameter count*.
주 언어는 Python, 실험은 Jupyter Notebook(`.ipynb`)으로 진행한다.

## 핵심 패러다임 — Training-time Dynamic Parameter Count

GraphLM 의 graph 는 **데이터를 graph 로 보는** (data-as-graph) 방향이 아니다.
또한 같은 computation-as-graph 안에서도, **활성 파라미터 (active) 만 동적** 인 sparse activation 류 (MoE 등) 는 본 프로젝트 1순위가 아니다.

| 구분 | Data-as-graph (NOT) | Sparse activation (NOT 1순위 — reference 만) | **Dynamic param count (THIS)** |
|---|---|---|---|
| 노드 정의 | 도메인 객체 (문서/분자/user) | FFN expert / attention head | **layer block / module 자체 (학습 중 추가·제거 가능 단위)** |
| 학습 중 총 파라미터 수 | 고정 | **고정** (활성만 변동) | **변동 (성장 / 축소 / sparse mask 진화)** |
| 학습 대상 | 노드 임베딩 | token → expert routing 정책 | **architecture growth operator / connection mask / module addition 정책** |
| 대표 갈래 | GCN, GAT, GraphSAGE, Graphormer | MoE, GShard, Switch, Mixtral, Mixture of Depths, Universal Transformer | **Net2Net, bert2BERT, LiGO, MSG (Growing) / SET, RigL (DST) / DARTS (NAS during training) / Progressive Stacking** |

이 패러다임 정의는 `~/.claude/projects/<...>/memory/project_paradigm.md` 에 영구 기록되어 있다 (2026-05-23 재정의 확정).

**시사점**:
- 신규 모델 / 실험은 **Growing / DST / DARTS** 우선. MoE 류는 baseline 비교용 reference.
- 신규 논문 큐레이션은 Net2Net / bert2BERT / LiGO / MSG / SET / RigL / DARTS / Stacking 계열 우선.
- 기존 `docs/papers/graph/` `docs/papers/hybrid/` 의 8편 + `docs/papers/computation-graph/` 의 8편 (sparse activation 위주) 은 **baseline reference 보존** (직접 채택 대상 아님).
- 모호한 작업은 dynamic count 쪽으로 보수 분류.

## 빠른 참조 (목차)

작업 유형별로 필요한 문서만 읽으세요. **전부 읽지 마세요.**

| 작업 | 참조 문서 | 핵심 내용 |
|------|-----------|-----------|
| 코드 스타일 확인 | `.claude/rules/06-code-style.md` | PEP 8, ruff, 네이밍, 한국어 커밋 |
| 에러 처리 | `.claude/rules/04-error-handling.md` | 예외 타입, 재시도, 로깅 필드 |
| 테스트 작성 | `.claude/rules/05-testing.md` | `tests/` 디렉토리 구조, 커버리지 70% |
| 아키텍처 이해 | `.claude/rules/01-architecture.md` | 모듈 레이아웃, 데이터 흐름, 노트북 구성 |
| **AI 작업 진행** | 공통 하네스 `~/.claude/rules/` + `.claude/rules/07-workflow.md` | **공통: 이슈 먼저 / 자율 진행 / commit-per-TODO / PR / 권한 / Label · Type / 결정 기록 / 오케스트레이션 — 전용: 스크립트 · 노트북 · loop.md** |
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

공통 규약 (이슈 먼저 · 자율 진행 경계 · commit-per-TODO · PR 자동 · 권한 최소화 · Label/Type · 결정 기록 · 오케스트레이션 · PR 피드백) 은
사용자 레벨 하네스 `~/.claude/CLAUDE.md` · `~/.claude/rules/` 가 기본값이다 (2026-10-11 이관, 이슈 #84). 프로젝트 규칙이 다르게 정하면 프로젝트가 우선한다.
본 저장소 전용 사항과 구 규약 번호 대응표는 [`.claude/rules/07-workflow.md`](.claude/rules/07-workflow.md). 요점:

- Label 은 `scripts/gh-meta.sh issue <N>` / `pr <N>` 으로 부여. Issue Type 은 `GRAPHLM_*_TYPE_ID` 환경변수가 설정된 경우에만 부여된다 (조회 방법은 스크립트 상단 주석)
- 노트북 (`.ipynb`) 은 `make nb-clean` 후 커밋
- 설계 변경 · 의사 결정은 PR 본문 "결정" 섹션에 ADR 형식으로 (전용 저장소 없음)
- PR 후 피드백은 세션 안에서 `Monitor` 로 처리하고 `scripts/pr-resolve-comments.sh` 로 완료 표시. `gh pr create` 직후 `.claude/loop.md` 를 cron 으로 자동 등록하던 동작은 폐지 — `loop.md` 는 사용자가 `/loop` 로 요청할 때만 쓰는 수동 보조 수단, `.claude/pr-feedback.md` 는 별도 자동화. 세션 종료 후 피드백은 사용자가 처리 (`@claude` 워크플로 없음)
- `~/.claude` 가 없는 환경 (CI) 에는 공통 규약이 로드되지 않는다 — 위 "반드시 지킬 규칙" 과 07-workflow.md 만 적용
