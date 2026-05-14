# GitHub Copilot Instructions

## Language
- All responses and code review comments MUST be written in **Korean (한국어)**
- Technical terms (Python, PyTorch, ruff, pytest 등) may remain in English
- Code itself (identifiers, docstrings) follows the conventions below

---

## Project Overview

**GraphLM** 은 그래프 구조와 언어 모델을 결합한 연구·실험 프로젝트입니다.
주 언어는 **Python**, 실험은 **Jupyter Notebook(`.ipynb`)** 으로 진행합니다.

```
src/graphlm/    → Python 패키지 (모델·그래프·유틸 구현)
notebooks/      → Jupyter 실험 노트북
tests/          → pytest 테스트 (src/graphlm/ 미러링)
data/           → 실험 데이터 (gitignore)
```

**Tech Stack**: Python 3.11+, ruff, pytest, Jupyter, PyTorch / JAX (실험에 따라 선택)

---

## Branch Naming Convention

```
{category}/#{issue-number}/{short-kebab-summary}
```

| Category | 용도 |
|----------|------|
| `feature/` | 새로운 기능 구현 및 추가 |
| `fix/` | 버그 수정 |
| `refactor/` | 구조 변경 및 리팩토링 |
| `docs/` | 문서·docstring·노트북 markdown |
| `chore/` | 빌드·CI·도구·의존성 |

**규칙:**
- 이슈 번호는 `#` 접두사 포함 (예: `#15`)
- 요약은 영문 소문자 + 하이픈(-) + 30자 이내

**예시:**
```
feature/#15/gcn-baseline-encoder
feature/#17/graph-attention-layer
fix/#7/nan-loss-detection
refactor/#4/train-config-frozen
docs/#2/api-docstrings
chore/#3/ruff-update
```

---

## Git Commit Convention

### Format

```
[{CATEGORY}]: {변경 내용}
```

### Categories

| Category | 용도 |
|----------|------|
| `FEAT` | 기능 구현 및 추가 |
| `FIX` | 버그 수정 |
| `REFAC` | 구조 변경, 리팩토링 |
| `DOCS` | 문서 작업, docstring, 노트북 markdown |
| `CHORE` | 빌드·CI·도구·의존성 등 잡무 |

### 작성 규칙

> **⚠️ 커밋 메시지는 반드시 한국어로 작성**

1. 언어: 한국어 (영어 사용 금지)
2. 형식: 명사형 종결 ("구현", "수정", "추가")
3. 내용: 변경 사항의 전체 요약 + 모듈별 변경점 명시

### 예시

```
[FEAT]: GCN baseline 인코더 구현

- GraphEncoder 클래스 추가 (in/hidden/out dim 설정 가능)
- 평균 이웃 집계 + GELU 활성화
- forward shape 보장 단위 테스트 5개 추가
```

```
[FIX]: 학습 루프에서 NaN loss 감지 시 즉시 중단

- 기존 silent skip 동작 제거
- ModelError 발생으로 원인 추적 가능
```

```
[REFAC]: TrainConfig 를 frozen dataclass 로 변경

- 학습 중 의도치 않은 config 변경 차단
- 호출부 3곳 갱신
```

```
[CHORE]: ruff 0.5.0 으로 업데이트

- pyproject.toml lint 규칙 ICN001 추가
- CI lint 버전 고정
```

```
[DOCS]: GraphEncoder docstring 보강 (en+ko)

- forward 의 입력/출력 shape 명시
- attention head 수에 따른 메모리 비용 안내
```

---

## Code Style Guide (Python)

### Formatting

- **Indentation**: 4 spaces (탭 금지)
- **Line Length**: 100자
- **Imports**: 표준 라이브러리 → 서드파티 → 로컬 (ruff 자동 정렬)

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from graphlm.data import GraphDataset
from graphlm.utils import set_seed
```

### Naming Conventions

| 대상 | 규칙 | 예시 |
|------|------|------|
| 변수·함수 | `snake_case` | `node_count`, `load_dataset` |
| 상수 | `UPPER_SNAKE` | `DEFAULT_BATCH_SIZE` |
| 클래스 | `PascalCase` | `GraphEncoder`, `TrainConfig` |
| 예외 | `PascalCase` + `Error` | `DataLoadError` |
| 모듈 | `snake_case` | `loader.py` |
| 패키지 | `snake_case` (단수) | `graphlm.data` |
| 비공개 | `_` prefix | `_internal_helper` |

### Function Design

- 함수당 최대 50줄
- 인자 5개 이상 → dataclass 로 묶기
- 모든 public 함수에 타입 힌트 필수
- Early return 패턴 선호

```python
# Good
def process(item: Item) -> Result:
    if not item.valid:
        return Result.empty()
    if item.cached:
        return item.cache
    return compute(item)
```

### Anti-Patterns (금지)

```python
# ❌ Magic numbers
if score > 0.85: ...
# ✅ Named constants
ACCEPT_THRESHOLD = 0.85
if score > ACCEPT_THRESHOLD: ...
```

```python
# ❌ Mutable default arg
def append(item, items=[]): ...
# ✅
def append(item, items=None):
    if items is None:
        items = []
    ...
```

```python
# ❌ bare except
try: risky()
except: pass
# ✅ 구체적 예외
try: risky()
except (ValueError, RuntimeError) as err:
    logger.warning("recovered", exc_info=err)
```

```python
# ❌ wildcard import
from graphlm.models import *
# ✅ 명시적 import
from graphlm.models import GraphEncoder
```

```python
# ❌ print 디버깅 잔재
print(f"DEBUG: {x.shape}")
# ✅ 정식 logging
logger.debug("forward", extra={"x_shape": tuple(x.shape)})
```

---

## Comments / Docstrings Convention

**언어 정책:**
- 인라인 주석: **한국어 우선**
- Docstring (exported 심볼): **영어 우선**, 한국어 설명 추가 가능
- 주석 원칙: **WHY만 설명** (WHAT은 코드로 표현)

```python
def aggregate_neighbors(x: Tensor, edge_index: Tensor) -> Tensor:
    """Aggregate neighbor features for each node.

    각 노드의 이웃 노드 feature 를 집계합니다.

    Args:
        x: Node features, shape ``[n_nodes, feat_dim]``.
        edge_index: COO format edges, shape ``[2, n_edges]``.

    Returns:
        Aggregated features, shape ``[n_nodes, feat_dim]``.
    """
    # softmax 전에 max 빼서 overflow 방지 (numerical stability)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    ...
```

**TODO 형식:**
```python
# TODO(juhy): edge weight 지원 추가 — 현재 unweighted 만 처리
# TODO: torch.compile() 적용 후 속도 비교
```

---

## Error Handling

- bare `except:` 금지 — 구체적 예외만
- `raise ... from err` 로 원인 chain 보존
- 사용자 정의 예외는 `GraphLMError` base 에서 파생

```python
# Good
try:
    dataset = load_dataset(path)
except FileNotFoundError as err:
    raise DataLoadError(f"데이터셋 로드 실패: {path}") from err
```

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Good — structured
logger.info("eval 완료", extra={
    "split": "test",
    "accuracy": 0.873,
    "n_samples": 1024,
})

# Bad — 문자열 보간
logger.info(f"eval 완료: acc=0.873, n=1024")
```

---

## Testing (pytest)

### Naming Pattern

```python
# test_{function}_{scenario}
def test_load_dataset_returns_expected_size(tmp_path): ...
def test_load_dataset_missing_file_raises(tmp_path): ...
```

### Structure (AAA)

```python
def test_load_dataset_returns_expected_size(tmp_path):
    # Arrange
    create_fake_dataset(tmp_path / "data.json", n=10)
    # Act
    ds = load_dataset(tmp_path / "data.json")
    # Assert
    assert len(ds) == 10
```

### Parametrize

```python
@pytest.mark.parametrize("input_, expected", [
    ("http://example.com", "https://example.com"),
    ("https://www.example.com", "https://example.com"),
])
def test_normalize_url(input_, expected):
    assert normalize_url(input_) == expected
```

### Coverage 기준

| 대상 | 최소 |
|---|---|
| 핵심 패키지 | 70% |
| `data/`, `graph/`, `eval/` | 90% |
| 예외 경로 | 100% |

### 위치

```
tests/                # src/graphlm/ 미러링
├── conftest.py
├── data/
├── models/
└── graph/
```

---

## Notebook 컨벤션

- 노트북에는 **분석 흐름 + 시각화만**. 로직은 `src/graphlm/` 에서 import.
- 셀 분리: 임포트 / 설정 / 데이터 / 모델 / 학습 / 평가
- 커밋 전 `make nb-clean` (또는 `nbstripout`) 으로 output cell 제거
- 노트북 상단 markdown 셀에 가설 / 데이터 / 시드 / 연관 이슈 명시

---

## Issue Convention

### 이슈 타이틀 형식

```
[{CATEGORY}] 이슈 타이틀
```

| Category | 라벨 | 템플릿 |
|----------|------|--------|
| `FEATURE` | `feature` | `feature.md` |
| `BUG` | `bug` | `bug.md` |
| `REFACTOR` | `refactor` | `refactor.md` |
| `CHORE` | `chore` | `chore.md` |

**예시:**
```
[FEATURE] GCN baseline 인코더 구현
[BUG] 학습 중 NaN loss 발생
[REFACTOR] TrainConfig 분리
[CHORE] ruff 버전 업데이트
```

### 이슈 ↔ 브랜치 ↔ PR ↔ 커밋 매핑

| 이슈 | 브랜치 prefix | 커밋/PR |
|------|---------------|---------|
| `FEATURE` | `feature/` | `FEAT` |
| `BUG` | `fix/` | `FIX` |
| `REFACTOR` | `refactor/` | `REFAC` |
| `CHORE` | `chore/` 또는 `docs/` | `CHORE` / `DOCS` |

---

## Pull Request Convention

### PR 타이틀 형식

CI (`PR Title Lint`) 가 정규식으로 엄격 강제합니다:
`^\[(FEAT|FIX|REFAC|DOCS|CHORE)#[0-9]+\]:? .+`

```
[{CATEGORY}#{이슈번호}] PR 타이틀
[{CATEGORY}#{이슈번호}]: PR 타이틀   (콜론 형태도 허용)
```

**통과 예시:**
```
[FEAT#15] GCN baseline 인코더 구현
[FIX#7] NaN loss 즉시 중단 처리
[REFAC#4] TrainConfig frozen 화
[DOCS#2] API docstring 동기화
[CHORE#3] ruff 0.5.0 업데이트
```

**거부 예시 (CI 실패):**
- `[FEAT]: 이슈번호 누락` — `#이슈번호` 필수
- `[FEAT 119]: # 대신 공백` — 반드시 `#` 사용
- `[FEAT#abc]: 숫자 아닌 이슈번호` — 숫자만 허용
- `[FEATXX#1]: 잘못된 카테고리` — 위 5개만 허용
- `feat#1: 소문자` — 대문자만 허용

### PR 라벨

| 라벨 | 적용 조건 |
|------|-----------|
| `feature` | FEAT 카테고리 PR |
| `bug` | FIX 카테고리 PR |
| `refactor` | REFAC 카테고리 PR |
| `documentation` | DOCS 카테고리 PR |
| `chore` | CHORE 카테고리 PR |
| `breaking change` | 하위 호환성 깨는 변경 |

---

## Code Review Checklist

PR 제출 전:

- [ ] 커밋 메시지가 한국어 + `[CATEGORY]:` 형식
- [ ] 브랜치명이 `{category}/#{issue}/{kebab}` 형식
- [ ] `ruff format .` 적용
- [ ] `ruff check .` 통과 + `nbqa ruff notebooks/` (있을 시)
- [ ] `pytest` 통과 + 커버리지 70%
- [ ] public 함수/클래스에 타입 힌트 + docstring
- [ ] 주석은 WHAT 이 아니라 WHY
- [ ] magic number 없음
- [ ] bare except 없음
- [ ] mutable default argument 없음
- [ ] print 디버깅 잔재 없음
- [ ] 노트북 commit 시 `make nb-clean` 적용
