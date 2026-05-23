# GraphLM

한국어 | _English (TBD)_

> **Transformer 내부 구조를 graph 로 바라보는** 연구·실험 프로젝트

## 개요

GraphLM 은 **Transformer 의 내부 sub-component (FFN expert, attention head, layer block 등) 를 그래프의 노드로 정의** 하고, token 임베딩이 그 노드 graph 위에서 routing / aggregation 되는 학습 framework 를 다룹니다.

> **핵심 패러다임**: **computation-as-graph** (Transformer 내부 구조 = 노드 graph)
> 데이터 객체 (문서/분자/user 등) 를 노드로 보는 **data-as-graph** 방향이 **아닙니다**. 일반 GCN/GAT/GraphSAGE 류는 baseline 비교 reference 로만 다룹니다.
>
> 대표 갈래: **MoE** (Switch Transformer, Mixtral), **Mixture of Depths**, **Graph HyperNetwork (GHN)**, NAS-Transformer.

주 언어는 Python, 실험은 Jupyter Notebook(`.ipynb`) 으로 진행합니다. 패러다임의 전체 정의는 [CLAUDE.md](../../CLAUDE.md#핵심-패러다임--computation-as-graph) 참조.

## 주요 특징

- 🐍 **Python 3.11+** — 타입 힌트 + dataclass 기반 설정
- 📓 **Jupyter 노트북 워크플로** — 가설 → 실험 → 분석 사이클
- 🧪 **재현성 우선** — 시드 고정, 의존성 버전 명시, 결과 저장 표준화
- 🧰 **단일 도구체인** — `ruff`(format + lint), `pytest`(테스트), `nbqa`(노트북 lint)
- 🤝 **AI 협업 친화적** — CLAUDE.md 기반 워크플로 (이슈-first / commit-per-TODO / 자동 PR)

## 디렉토리 구조

```
GraphLM/
├── src/graphlm/        # Python 패키지 (재사용 가능한 모듈)
│   ├── data/           # Dataset, Loader, 전처리
│   ├── models/         # 모델 정의
│   ├── training/       # 학습 루프
│   ├── eval/           # 평가 메트릭
│   ├── graph/          # 그래프 유틸
│   ├── utils/          # 시드, 로깅
│   └── config/         # 실험 설정
├── notebooks/          # Jupyter 실험 노트북
│   ├── 00-exploration/
│   ├── 10-experiments/
│   └── 20-analysis/
├── tests/              # pytest (src/graphlm/ 미러링)
├── data/               # 실험 데이터 (gitignore)
├── docs/               # CI / 한국어 문서
└── .claude/rules/      # 개발 규칙
```

상세는 [.claude/rules/01-architecture.md](../../.claude/rules/01-architecture.md) 참조.

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- (선택) GPU (CUDA 11.8+)

### 설치 (uv 기반)

```bash
git clone <repo-url>
cd GraphLM

# .venv 생성 + dev/notebook 의존성 + editable 설치 + Jupyter 커널 등록
make install-notebook
make kernel
```

`make kernel` 실행 후 Jupyter 의 커널 선택 메뉴에 **GraphLM (uv .venv)** 항목이 나타납니다.
새 노트북 생성 시 이 커널을 선택하면 `.venv` 의 패키지를 그대로 import 할 수 있습니다.

### 자주 쓰는 명령

```bash
make venv             # .venv 생성만
make install          # .venv + dev 의존성 (노트북 제외)
make install-notebook # .venv + dev + notebook 의존성
make kernel           # .venv 를 Jupyter 커널로 등록
make test             # pytest 실행 (uv run)
make coverage         # 커버리지 리포트
make lint             # ruff check + nbqa
make fmt              # ruff format
make nb-clean         # 노트북 output cell 정리
make help             # 모든 타겟 표시
```

> `make` 타겟은 내부적으로 `uv run` 으로 `.venv` 를 사용합니다.
> `source .venv/bin/activate` 로 활성화 후 직접 명령을 실행해도 됩니다.

### 실험 실행

```bash
# Jupyter 서버 기동 (커널 선택: "GraphLM (uv .venv)")
uv run jupyter lab

# 또는 노트북을 스크립트로 실행
uv run jupyter nbconvert --to notebook --execute notebooks/10-experiments/<파일>.ipynb
```

### VS Code / Cursor 에서 노트북 열기

1. 노트북 (`.ipynb`) 파일 우상단의 **커널 선택** 클릭
2. **GraphLM (uv .venv)** 선택
3. 셀 실행 시 `.venv` 의 `graphlm` 패키지가 import 됨

## 개발 워크플로 (AI 협업 포함)

본 프로젝트는 AI(Claude, Copilot 등)와의 협업을 전제로 설계되었습니다.
모든 워크플로 규약은 [.claude/rules/07-workflow.md](../../.claude/rules/07-workflow.md) 에서 정의합니다.

핵심 6 규약:

1. **이슈 먼저** — 코드 수정 전 GitHub 이슈 생성
2. **자율 진행** — destructive / 시스템 / 외부 영향만 사용자 확인
3. **Commit-per-TODO** — 논리적 변경 단위마다 commit
4. **PR 자동 생성** — 작업 완료 직후 컨벤션 준수 PR 생성
5. **권한 최소화** — 새 의존성 / 도구는 명시적 사전 확인
6. **Label · Type 부여** — `scripts/gh-meta.sh` 로 자동 부여

## 컨벤션

### 커밋 메시지

```
[FEAT]: 한국어 변경 내용
[FIX]: ...
[REFAC]: ...
[DOCS]: ...
[CHORE]: ...
```

### PR 타이틀

```
[FEAT#15] GCN baseline 인코더 구현
[FIX#7] NaN loss 즉시 중단 처리
```

### 이슈 타이틀

```
[FEATURE] GCN baseline 인코더
[BUG] NaN loss 발생
[REFACTOR] TrainConfig 분리
[CHORE] ruff 0.5.0 업데이트
[DOCS] API docstring 동기화
[HOTFIX] eval 시 dim mismatch
```

상세는 [.github/copilot-instructions.md](../../.github/copilot-instructions.md) 참조.

## CI / 머지 게이트

| 게이트 | 도구 |
|---|---|
| 커밋 메시지 형식 | regex (Commit Lint) |
| PR 타이틀 형식 | regex (PR Title Lint) |
| Linked Issue | GraphQL (Linked Issue Check) |
| 포맷 | `ruff format --check` |
| 빌드 | `pip install -e ".[dev]"` + import |
| 테스트 | `pytest --cov --cov-fail-under=70` |
| 정적 분석 | `ruff check` + `nbqa ruff` |

상세는 [docs/ci/conventions.md](../ci/conventions.md) 와
[docs/ci/status-checks.md](../ci/status-checks.md) 참조.

## 기여

기여하기 전에 [.claude/rules/](../../.claude/rules/) 의 개발 규칙을 읽어주세요.

## 라이선스

TBD.
