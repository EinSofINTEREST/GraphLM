# Testing and Quality Assurance Rules

## Testing Philosophy

연구 코드라도 다음은 테스트한다:

- **데이터 처리 (`src/graphlm/data/`)** — 전처리·split·collate 함수
- **그래프 유틸 (`src/graphlm/graph/`)** — 그래프 구성·변환 함수
- **모델 forward shape** — 입력 shape → 출력 shape 보장
- **메트릭 (`src/graphlm/eval/`)** — 계산식 검증 (known input → known output)
- **설정 로더 (`src/graphlm/config/`)** — 잘못된 설정은 명시적 예외

**노트북은 테스트 대상이 아니다** — 노트북에 들어갈 로직은 패키지 모듈로 추출해서 테스트.

## Coverage 기준

| 대상 | 최소 커버리지 |
|---|---|
| 핵심 패키지 (전체 평균) | 70% |
| `data/`, `graph/`, `eval/` (deterministic 함수) | 90% |
| 예외 처리 경로 | 100% |
| 모델 forward / 학습 루프 | shape · 단일 step 작동 정도만 (full convergence 테스트 불필요) |

## 테스트 위치

```
tests/                         # 패키지 본체와 분리
├── conftest.py               # 공통 fixture (시드, tmp 데이터)
├── data/                     # ← src/graphlm/data/ 테스트
│   └── test_loader.py
├── graph/                    # ← src/graphlm/graph/
│   └── test_builder.py
├── models/
│   └── test_encoder.py
├── eval/
│   └── test_metrics.py
└── utils/
    └── test_seed.py
```

- 디렉토리 구조는 `src/graphlm/` 미러링
- 파일명: `test_<module>.py`
- 함수명: `test_<function>_<scenario>` (예: `test_load_dataset_missing_file_raises`)

## pytest 사용

### 기본 패턴

```python
# tests/data/test_loader.py
import pytest

from graphlm.data import load_dataset
from graphlm.utils.exceptions import DataLoadError


def test_load_dataset_returns_expected_size(tmp_path):
    # Arrange
    create_fake_dataset(tmp_path / "data.json", n=10)

    # Act
    ds = load_dataset(tmp_path / "data.json")

    # Assert
    assert len(ds) == 10


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError, match="경로 없음"):
        load_dataset(tmp_path / "nonexistent.json")
```

### Parametrize (table-driven tests)

```python
@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("http://example.com", "https://example.com"),
        ("https://www.example.com", "https://example.com"),
        ("https://example.com/path/", "https://example.com/path"),
    ],
)
def test_normalize_url(input_str, expected):
    assert normalize_url(input_str) == expected


@pytest.mark.parametrize("bad_input", ["", "not-a-url", None])
def test_normalize_url_invalid_raises(bad_input):
    with pytest.raises(ValueError):
        normalize_url(bad_input)
```

### Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def fix_seed():
    """모든 테스트에 동일 시드 적용."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def small_graph():
    """3-node toy graph (테스트용)."""
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    features = torch.randn(3, 8)
    return {"edges": edges, "features": features}
```

### 마커

느린 / GPU 필요 / 외부 네트워크 의존 테스트는 마커로 분리:

```python
@pytest.mark.slow
def test_full_epoch_runs():
    ...

@pytest.mark.gpu
def test_cuda_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA 미사용")
    ...

@pytest.mark.network
def test_hf_hub_download():
    ...
```

`pyproject.toml` 에 마커 등록:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: 1초 이상 걸리는 테스트",
    "gpu: CUDA 필요",
    "network: 외부 네트워크 의존",
]
```

CI 에서는 기본적으로 `slow` / `gpu` / `network` 마커 제외:

```bash
pytest -m "not slow and not gpu and not network"
```

## 모델 / 텐서 테스트 패턴

### Shape 보장

```python
def test_encoder_output_shape():
    model = GraphEncoder(in_dim=16, hidden_dim=32, out_dim=8)
    n_nodes = 10
    x = torch.randn(n_nodes, 16)
    edge_index = torch.randint(0, n_nodes, (2, 20))

    out = model(x, edge_index)

    assert out.shape == (n_nodes, 8)
```

### Determinism

```python
def test_forward_is_deterministic():
    torch.manual_seed(0)
    model = GraphEncoder(16, 32, 8)
    x = torch.randn(10, 16)
    edge_index = torch.randint(0, 10, (2, 20))

    torch.manual_seed(0)
    out_a = model(x, edge_index)
    torch.manual_seed(0)
    # 재초기화로 동일 weight 보장
    model = GraphEncoder(16, 32, 8)
    out_b = model(x, edge_index)

    assert torch.allclose(out_a, out_b)
```

### Gradient flow

```python
def test_loss_backward_runs():
    model = GraphEncoder(16, 32, 8)
    x = torch.randn(10, 16, requires_grad=False)
    edge_index = torch.randint(0, 10, (2, 20))

    out = model(x, edge_index)
    loss = out.sum()
    loss.backward()

    # 학습 가능한 모든 파라미터에 gradient 가 흘렀는지 확인
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} 에 grad 없음"
```

## 외부 의존성 모킹

### 파일 시스템

`tmp_path` fixture 사용 (자동 cleanup):

```python
def test_save_load_roundtrip(tmp_path):
    config = TrainConfig(batch_size=32, lr=1e-3)
    path = tmp_path / "config.yaml"

    config.save(path)
    loaded = TrainConfig.load(path)

    assert loaded == config
```

### HTTP / 네트워크

`pytest-httpx` 또는 `responses` 사용. 실제 네트워크는 `@pytest.mark.network` 로 분리.

### Hugging Face Hub

테스트에서 실제 모델 다운로드 금지 — 작은 toy 모델 또는 모의 객체 사용.

## Notebook 검증

노트북은 단위 테스트 대상이 아니지만, **실행 가능 여부** 는 검증 가치 있음.
CI 에서 `nbqa ruff notebooks/` 로 정적 검사만 수행. notebook 실행 테스트는 비용·flakiness 때문에 기본 CI 에서 제외.

필요 시 `pytest --nbmake notebooks/` 로 로컬에서 실행 검증 가능.

## 커버리지

### 명령

```bash
# 커버리지 측정
pytest --cov=graphlm --cov-report=term-missing --cov-report=html

# 최소 커버리지 강제
pytest --cov=graphlm --cov-fail-under=70
```

### 설정

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/graphlm"]
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## CI 통합

GitHub Actions 의 `Test` job 이 `pytest -m "not slow and not gpu and not network" --cov=graphlm --cov-fail-under=70` 으로 실행.
세부는 [`docs/ci/conventions.md`](../../docs/ci/conventions.md) 참조.
