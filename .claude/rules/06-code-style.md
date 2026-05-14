# Code Style and Conventions

## Core Principles

- **PEP 8 준수** — `ruff` 가 자동 강제
- **가독성 우선** — 영리한 코드보다 명확한 코드
- **최소 주석** — 코드가 스스로 설명. 주석은 *WHY* 만
- **과도한 추상화 금지** — 연구 코드는 단순해야 한다. 호출자 1명짜리 추상화는 inline

## Python Style

### 포맷팅

- **포맷터**: `ruff format` (Black 호환)
- **들여쓰기**: 4 spaces
- **라인 길이**: 100자 (ruff 기본 88 → 100 으로 완화)
- **임포트 순서**: 표준 → 서드파티 → 로컬 (ruff 가 자동 정렬)

```python
# 1. 표준 라이브러리
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# 2. 서드파티
import numpy as np
import torch
from torch import Tensor

# 3. 로컬
from graphlm.data import GraphDataset
from graphlm.utils import set_seed
```

### 네이밍

| 대상 | 규칙 | 예시 |
|---|---|---|
| 변수·함수 | `snake_case` | `node_count`, `load_dataset` |
| 상수 | `UPPER_SNAKE` | `DEFAULT_BATCH_SIZE`, `MAX_EPOCHS` |
| 클래스 | `PascalCase` | `GraphEncoder`, `TrainConfig` |
| 사용자 예외 | `PascalCase` + `Error` 접미사 | `DataLoadError`, `ConfigError` |
| 모듈 | `snake_case` (짧게) | `graph.py`, `loader.py` |
| 패키지 | `snake_case` (단수) | `graphlm.data`, `graphlm.eval` |
| 비공개 | `_` prefix | `_internal_helper` |
| dunder | `__init__`, `__repr__` 등 표준만 | — |

### 타입 힌트

**모든 public 함수 / 메서드에 타입 힌트 필수**. 내부 helper 는 선택.

```python
# Good
def normalize_features(
    x: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# Bad — 타입 없음
def normalize_features(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)
```

복잡한 타입은 alias:

```python
from typing import TypeAlias

NodeFeatures: TypeAlias = Tensor          # shape: [n_nodes, feat_dim]
EdgeIndex: TypeAlias = Tensor             # shape: [2, n_edges]
GraphBatch: TypeAlias = dict[str, Tensor]
```

### 함수 설계

- **함수당 최대 50줄** (학습 루프 등 예외 허용 — 분리 가능하면 분리)
- **인자 5개 이상 → dataclass 로 묶기**
- **Early return** 선호

```python
# Good
def process(item: Item) -> Result:
    if not item.valid:
        return Result.empty()
    if item.cached:
        return item.cache
    return compute(item)

# Bad — else 중첩
def process(item: Item) -> Result:
    if item.valid:
        if item.cached:
            return item.cache
        else:
            return compute(item)
    else:
        return Result.empty()
```

### Dataclass / Config

설정은 `@dataclass` 또는 `pydantic.BaseModel` 로:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # data
    data_dir: Path
    batch_size: int = 32

    # model
    hidden_dim: int = 128
    n_layers: int = 3

    # optim
    lr: float = 1e-3
    weight_decay: float = 0.0

    # runtime
    seed: int = 42
    device: str = "cuda"
    max_epochs: int = 100
```

`frozen=True` 로 의도치 않은 변경 차단. 그룹별 빈 줄로 가독성.

## 클래스 설계

### 단일 책임

```python
# Good — 모델만 책임
class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        ...

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        ...

# Bad — 모델이 데이터 로드까지
class GraphEncoder(nn.Module):
    def __init__(self, data_path: Path):
        super().__init__()
        self.dataset = load_dataset(data_path)  # ❌
        ...
```

### `__repr__` 제공

디버깅 / 로그에 도움. dataclass 는 자동, nn.Module 도 자동, 직접 작성하는 클래스는 명시.

## 주석 / Docstring

### 언어 정책

- **Docstring**: 영어 우선, 한국어 보조 허용. public 심볼은 docstring 필수.
- **인라인 주석**: 한국어 우선 (디버깅 시 사용자가 읽음).
- **금지**: WHAT 주석 (코드를 읽으면 알 수 있는 내용).

### Docstring 스타일

Google style 권장 (ruff 가 PEP257 기본 검사):

```python
def aggregate_neighbors(
    x: Tensor,
    edge_index: Tensor,
    *,
    method: str = "mean",
) -> Tensor:
    """Aggregate neighbor features for each node.

    각 노드의 이웃 노드 feature 를 집계합니다.

    Args:
        x: Node features, shape ``[n_nodes, feat_dim]``.
        edge_index: Edge index in COO format, shape ``[2, n_edges]``.
        method: Aggregation method. One of ``"mean"``, ``"sum"``, ``"max"``.

    Returns:
        Aggregated features, shape ``[n_nodes, feat_dim]``.

    Raises:
        ValueError: If ``method`` is not a supported aggregation.
    """
    ...
```

### 인라인 주석 — WHY 만

```python
# Good — 비자명한 이유 설명
# softmax 전에 max 빼서 overflow 방지 (numerical stability)
scores = scores - scores.max(dim=-1, keepdim=True).values
attn = scores.softmax(dim=-1)

# Bad — 코드가 이미 설명
# scores 에서 max 를 뺀다
scores = scores - scores.max(dim=-1, keepdim=True).values
```

### TODO 주석

```python
# TODO(juhy): edge weight 지원 추가 — 현재 unweighted 만 처리
# TODO: torch.compile() 적용 후 속도 비교 (현재 약 1.5x slow)
```

작성자 / 컨텍스트 포함. 단순 `# TODO` 금지.

## Notebook 컨벤션

### 셀 구조 권장

```
[셀 1] 제목 + 가설 (markdown)
[셀 2] 임포트 + 시드 + device 설정
[셀 3] 설정 (TrainConfig dataclass 인스턴스)
[셀 4~N] 데이터 로드 / 모델 정의 / 학습 / 평가 (각 1셀)
[마지막 셀] 결과 요약 + 다음 가설 (markdown)
```

### 셀 안에서

- **로직 정의 금지** — 함수 / 클래스는 `src/graphlm/` 에 작성하고 import
- **하드코딩 금지** — 하이퍼파라미터는 config 셀에 모음
- **출력 정리** — 큰 tensor `print()` 지양, 통계만 출력

### 노트북 메타데이터

상단 markdown 셀에:

```markdown
# 10-baseline-gnn-vs-gcn

- 가설: GCN baseline 대비 attention 기반 GNN 이 small graph 에서 우위
- 데이터: Cora, Citeseer
- 시드: 42
- 작성일: 2026-05-14
- 연관 이슈: #15
```

## Anti-Patterns (금지)

### 1. Magic numbers

```python
# Bad
if score > 0.85:
    accept()
# Good
ACCEPT_THRESHOLD = 0.85
if score > ACCEPT_THRESHOLD:
    accept()
```

### 2. 깊은 중첩

```python
# Bad
if a:
    if b:
        if c:
            if d:
                do_something()

# Good
if not (a and b and c and d):
    return
do_something()
```

### 3. mutable default argument

```python
# Bad — list 가 함수 호출 간 공유됨
def append_item(item, items=[]):
    items.append(item)
    return items

# Good
def append_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 4. bare except

```python
# Bad
try:
    risky()
except:
    pass

# Good — 구체적 예외만
try:
    risky()
except (ValueError, RuntimeError) as err:
    logger.warning("recovered", exc_info=err)
```

### 5. 불필요한 변수

```python
# Bad
def is_valid(s: str) -> bool:
    result = len(s) > 0
    return result

# Good
def is_valid(s: str) -> bool:
    return len(s) > 0
```

### 6. print 디버깅 잔재

```python
# Bad — 커밋에 남기지 말 것
print(f"DEBUG: x.shape = {x.shape}")

# Good — 정식 로깅
logger.debug("forward", extra={"x_shape": tuple(x.shape)})
```

### 7. wildcard import

```python
# Bad
from graphlm.models import *

# Good
from graphlm.models import GraphEncoder, GraphAttention
```

## Git Commit Conventions

### Format

```
[{카테고리}]: {변경 내용}
```

### 카테고리

- **FEAT**: 기능 구현 및 추가
- **FIX**: 버그 수정
- **REFAC**: 구조 변경, 리팩토링
- **DOCS**: 문서·docstring·주석·노트북 markdown 셀
- **CHORE**: 빌드·CI·도구·의존성

### 변경 내용 작성 규칙

**⚠️ 모든 커밋 메시지는 한국어로 작성.**

1. 언어: 한국어
2. 형식: 명사형 종결 ("구현", "수정", "추가")
3. 내용: 전체 요약 + 모듈별 변경점 명시

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
- ModelError 발생으로 원인 추적 가능하게 변경
- 회귀 테스트 추가
```

```
[REFAC]: TrainConfig 를 frozen dataclass 로 변경

- 학습 중 의도치 않은 config 변경 차단
- 호출부 3곳 갱신
```

```
[DOCS]: GraphEncoder docstring 보강 (en+ko)

- forward 의 입력/출력 shape 명시
- attention head 수에 따른 메모리 비용 안내
```

```
[CHORE]: ruff 0.5.0 으로 업데이트

- pyproject.toml lint 규칙 ICN001 추가
- CI lint 버전 고정
```

### Branch Naming

```
{카테고리}/#{이슈번호}/{핵심-변경-대상-요약}
```

| 카테고리 | 용도 |
|---|---|
| `feature/` | 새 기능 |
| `fix/` | 버그 수정 |
| `refactor/` | 구조 변경 |
| `docs/` | 문서 |
| `chore/` | 잡무 |

영문 소문자 + 하이픈, 30자 이내.

예시:

```
feature/#15/gcn-baseline-encoder
feature/#17/graph-attention-layer
fix/#7/nan-loss-detection
refactor/#4/train-config-frozen
docs/#2/api-docstrings
chore/#3/ruff-update
```

## Database / Config 파일 (참고)

연구 단계라 DB 사용은 적지만, 사용 시:

### YAML / TOML

- 들여쓰기: 2 spaces
- 키: `snake_case`
- 비자명 값은 단위 / 의미 주석

```toml
[train]
batch_size = 32
lr = 1e-3
max_epochs = 100
warmup_steps = 1000   # learning rate warmup duration

[data]
num_workers = 4       # DataLoader 워커 수
pin_memory = true     # GPU 학습 시 권장
```

## 파일 구조 (모듈 내부)

```python
"""Module docstring — 모듈 목적 한 줄."""

from __future__ import annotations

# 1. 표준 라이브러리
import logging
from pathlib import Path

# 2. 서드파티
import torch
from torch import Tensor, nn

# 3. 로컬
from graphlm.utils import set_seed

# 4. 모듈 상수
logger = logging.getLogger(__name__)
DEFAULT_HIDDEN_DIM = 128

# 5. Public API (클래스 / 함수)
class GraphEncoder(nn.Module):
    ...

def aggregate(x: Tensor, edge_index: Tensor) -> Tensor:
    ...

# 6. Private helpers
def _normalize(x: Tensor) -> Tensor:
    ...
```

## Code Review Checklist

PR 제출 전:

- [ ] 커밋 메시지가 한국어 + `[카테고리]:` 형식
- [ ] 브랜치명이 `{카테고리}/#{이슈}/{kebab}` 형식
- [ ] `ruff format .` 적용
- [ ] `ruff check .` 통과
- [ ] `pytest` 통과 + 커버리지 70%
- [ ] public 함수 / 클래스에 타입 힌트 + docstring
- [ ] 주석은 WHAT 이 아니라 WHY
- [ ] magic number 없음
- [ ] bare except 없음
- [ ] mutable default arg 없음
- [ ] 노트북 commit 시 `make nb-clean` 적용
- [ ] print 디버깅 잔재 없음
