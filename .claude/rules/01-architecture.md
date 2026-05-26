# Architecture and Design Principles

## Core Architecture

### 프로젝트 성격

GraphLM 은 **학습 중 Transformer 의 파라미터 수를 동적으로 조정** 하기 위해 모델 구조를 graph 로 다루는 연구·실험 프로젝트입니다.
production 서비스가 아니라 reproducible research codebase 를 지향합니다.

**핵심 패러다임 (필수 인지)**:

- **목적** = training-time dynamic parameter count (학습 중 총 파라미터 수 변동)
- **노드** = layer block / module / connection 등 추가·제거·재활성화 가능한 단위
- **Edge** = module 간 데이터 흐름 / 활성화 mask
- **학습 대상** = architecture growth operator / connection mask 진화 / module addition 정책
- **대표 갈래** =
  - **Growing Networks** (function-preserving expansion): Net2Net, bert2BERT, LiGO (ICLR 2023), MSG
  - **Progressive Stacking**: Stacking BERT (ICML 2019)
  - **Dynamic Sparse Training (DST)**: SET, RigL (ICML 2020)
  - **Differentiable Architecture Search**: DARTS (ICLR 2019)

본 프로젝트가 다루지 **않는** 두 방향:
- **Data-as-graph** (문서/분자/user 를 노드로 보는 GCN/GAT/GraphSAGE 류) — `docs/papers/graph/`, `hybrid/` 의 8편은 baseline reference 만.
- **Sparse activation with fixed total** (MoE / GShard / Switch / Mixtral / Mixture of Depths / Universal Transformer) — `docs/papers/computation-graph/` 의 6편은 baseline reference 만. 활성 파라미터만 동적이지 총 파라미터는 고정이라 본 프로젝트 목적과 다름.

`src/graphlm/models/` 의 1순위 구현은 **growing Transformer** (예: LiGO 류 학습 가능 expansion operator, Progressive Stacking 의 layer 점진 추가) 입니다. 전체 정의는 [`CLAUDE.md`](../../CLAUDE.md#핵심-패러다임--training-time-dynamic-parameter-count) 참조.

설계 우선순위:

1. **재현성 (Reproducibility)** — 동일 시드 / 동일 환경 / 동일 데이터 → 동일 결과
2. **모듈성 (Modularity)** — 모델·데이터·실험 루프 분리. 노트북에서 import 해 사용 가능
3. **단순성 (Simplicity)** — 과도한 추상화 금지. 연구 코드는 읽기 쉬워야 한다
4. **점진성 (Iterability)** — 가설 → 노트북 실험 → 패키지화 → 재실험 사이클

## 코드 / 노트북 분리 원칙

| 종류 | 위치 | 용도 |
|---|---|---|
| **패키지 코드** | `src/graphlm/` | 재사용 가능한 모듈 — 모델·데이터로더·메트릭·유틸 |
| **노트북** | `notebooks/` | 실험 진행 / 시각화 / 분석. **로직은 패키지에서 import** |
| **테스트** | `tests/` | pytest 테스트 — 패키지 코드만 대상 |
| **스크립트** | `scripts/` | CI / 도구 / 일회성 실행 스크립트 |

**규칙**: 노트북에는 분석 흐름 + 시각화만, 핵심 로직은 `src/graphlm/` 모듈로 추출.
"노트북 안에서 함수 정의 → 다른 노트북에서 복붙" 패턴은 금지.

## 디렉토리 구조

```
GraphLM/
├── src/
│   └── graphlm/                # Python 패키지 (import graphlm)
│       ├── __init__.py
│       ├── data/               # Dataset, Loader, 전처리
│       ├── models/             # 모델 정의 (graph encoder, LM head 등)
│       ├── training/           # 학습 루프, optimizer, scheduler
│       ├── eval/               # 평가 메트릭, 벤치마크
│       ├── graph/              # 그래프 구성·조작 유틸
│       ├── utils/              # 로깅, 시드, 설정 로더
│       └── config/             # 실험 설정 (dataclass 또는 hydra/omegaconf)
│
├── notebooks/                  # Jupyter 실험 노트북
│   ├── 00-exploration/         # 데이터 탐색
│   ├── 01-experiments/         # 가설 검증 실험 (block-level paradigm)
│   ├── 02-function-level/      # neuron paradigm 실험 (head-level)
│   ├── 03-analysis/            # 결과 분석·시각화
│   └── README.md               # 노트북 명명 규칙·인덱스
│
├── tests/                      # pytest (src/graphlm/ 미러링)
│   ├── data/
│   ├── models/
│   ├── graph/
│   └── conftest.py
│
├── data/                       # 실험 데이터 (gitignore)
│   ├── raw/                    # 원본 (다운로드 후 변경 금지)
│   ├── processed/              # 전처리 산출물
│   └── README.md               # 데이터 출처 / 라이선스
│
├── scripts/                    # CI / 운영 / 유틸 스크립트
│   ├── lint.sh
│   ├── gh-meta.sh
│   └── ...
│
├── docs/
│   ├── ci/                     # CI 운영 규약
│   ├── ko/                     # 한국어 문서
│   └── prompts/                # AI 프롬프트 템플릿
│
├── .claude/rules/              # 개발 규칙 (목차: ../../CLAUDE.md)
├── .github/                    # 워크플로 / 이슈·PR 템플릿 / CODEOWNERS
├── pyproject.toml              # 패키지 메타데이터 + ruff/pytest/mypy 설정
├── Makefile                    # install / test / lint / fmt / nb-clean
├── README.md
└── CLAUDE.md
```

### 디렉토리 목적

**`src/graphlm/`**: 패키지 본체. `pip install -e .` 으로 editable 설치 후 노트북·테스트에서 `from graphlm import ...` 으로 import.

**`notebooks/`**: 실험 노트북. 번호 prefix(`00-/10-/20-`)로 단계 구분. 노트북 파일명은 `<번호>-<주제>.ipynb` (예: `10-baseline-gnn-vs-gcn.ipynb`).

**`tests/`**: 패키지 코드 단위 테스트. 노트북은 테스트 대상이 아님 (노트북은 실험 산출물, 패키지가 검증 대상).

**`data/`**: gitignore. 큰 데이터는 외부 스토리지 / DVC / Hugging Face Datasets 권장. raw 는 read-only.

## 데이터 흐름 (실험 사이클)

```
[가설] → notebooks/00-exploration/ 에서 데이터 탐색
       → src/graphlm/data/ 에 dataset 클래스 작성
       → src/graphlm/models/ 에 모델 정의
       → notebooks/01-experiments/ 에서 학습·평가 실행 (또는 notebooks/02-function-level/)
       → src/graphlm/eval/ 메트릭 함수로 결과 측정
       → notebooks/03-analysis/ 에서 결과 분석·시각화
       → [재가설] 또는 [논문/리포트]
```

## 기술 스택

### 코어

- **Python**: 3.11+
- **패키지 관리**: `uv` 권장 (또는 `pip` + `pyproject.toml`)
- **포맷터/린터**: `ruff` (포맷 + lint 단일 도구)
- **노트북 린트**: `nbqa` (notebook 에 ruff 적용)
- **노트북 정리**: `nbstripout` (커밋 전 output cell 제거)
- **테스트**: `pytest`, `pytest-cov`
- **타입 체크 (선택)**: `mypy`

### 도메인 라이브러리 (예시 — 실제 채택은 실험 진행하며 결정)

연구 진행에 따라 선택하는 도메인 의존성은 별도 의존성 추가 PR 로 관리:

- ML 프레임워크: `torch` / `jax` 중 택 1
- 그래프: `torch-geometric` / `dgl` / `networkx`
- LM: `transformers` (Hugging Face)
- 시각화: `matplotlib`, `seaborn`, `plotly`
- 데이터: `pandas`, `numpy`, `datasets`

신규 의존성 추가는 [07-workflow.md](07-workflow.md) 의 권한 최소화 정책에 따라 명시 PR 로 분리한다.

## 재현성 원칙

1. **고정 시드** — 모든 실험은 시드 명시 (random / numpy / torch). `graphlm.utils.set_seed(seed)` 권장.
2. **버전 명시** — 의존성은 `pyproject.toml` 에 정확 또는 범위 버전 명시.
3. **설정 분리** — 하이퍼파라미터는 코드 하드코딩 X. dataclass 또는 yaml/toml 설정으로 분리.
4. **결과 저장** — 메트릭 / 로그 / 체크포인트는 `runs/<실험명>/` 아래로 저장 (gitignore). 실험명에 시드·날짜·주요 hp 포함.
5. **노트북 출력 정리** — 커밋 전 `make nb-clean` 으로 output cell 제거 (diff 폭주 방지). 필요한 figure 는 `docs/` 또는 별도 export.

## 성능 / 자원 고려 (연구 단계)

연구 단계에서는 production 수준 SLA 가 없으므로 다음 원칙만 지킴:

- **단일 머신 기준** — 분산 학습은 명백한 필요 발생 후 도입
- **메모리 인지** — 큰 데이터는 streaming / chunked loading
- **GPU 사용 시 명시** — 노트북 상단에 `device = "cuda" if torch.cuda.is_available() else "cpu"` 명시
- **장시간 학습은 백그라운드 실행 + 체크포인트** — 노트북 셀로 8시간 학습 돌리지 말 것
