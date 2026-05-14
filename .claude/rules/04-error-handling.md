# Error Handling and Logging Rules

연구 코드는 production 코드만큼 강건한 에러 핸들링을 요구하지 않지만,
**디버깅 시간 단축 / 재현 실패 원인 파악**을 위해 다음 원칙은 지킨다.

## 핵심 원칙

1. **빨리 실패 (Fail Fast)** — 잘못된 입력이나 환경은 즉시 예외. 조용히 `None` 반환 금지.
2. **컨텍스트 보존** — `raise ... from err` 로 원인 chain 유지. bare `except:` 금지.
3. **사용자 정의 예외** — 의미 있는 카테고리는 자체 예외 클래스 정의.
4. **로깅** — print 대신 `logging` 모듈. 구조화된 필드.

## 예외 사용 원칙

### bare except 금지

```python
# Bad — 모든 예외를 삼킴 (KeyboardInterrupt 까지 포함)
try:
    train(model)
except:
    pass

# Bad — 너무 광범위
try:
    train(model)
except Exception:
    pass

# Good — 구체적인 예외만
try:
    train(model)
except RuntimeError as err:
    logger.exception("학습 실패")
    raise
```

### 예외 wrapping (chain 보존)

```python
# Good — 원인 보존
try:
    dataset = load_dataset(path)
except FileNotFoundError as err:
    raise DataLoadError(f"데이터셋 로드 실패: {path}") from err

# Bad — 원인 손실
try:
    dataset = load_dataset(path)
except FileNotFoundError:
    raise DataLoadError("데이터셋 로드 실패")
```

### 사용자 정의 예외

도메인별 의미 있는 카테고리는 자체 예외 정의:

```python
# src/graphlm/utils/exceptions.py
class GraphLMError(Exception):
    """GraphLM 패키지의 모든 사용자 정의 예외 base."""

class DataLoadError(GraphLMError):
    """데이터 로드 실패."""

class ConfigError(GraphLMError):
    """잘못된 실험 설정."""

class ModelError(GraphLMError):
    """모델 정의/forward 단계 오류."""

class ReproducibilityError(GraphLMError):
    """시드 고정 실패 / 결정론 깨짐 감지."""
```

base 클래스가 있으면 호출자가 `except GraphLMError` 로 라이브러리 예외만 분리 가능.

## 재시도

연구 코드에서 retry 가 필요한 경우는 제한적 — 주로 외부 API/네트워크 (예: HF Hub 다운로드).

`tenacity` 라이브러리 권장:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def download_dataset(url: str) -> Path:
    ...
```

학습 루프 안의 NaN/Inf 감지 같은 "재시도하면 의미 없는" 케이스는 retry 대신 즉시 fail.

## 입력 검증

함수 경계에서 입력 검증 — 특히 외부 입력 (config, file, CLI 인자):

```python
def train(config: TrainConfig) -> None:
    if config.batch_size <= 0:
        raise ConfigError(f"batch_size 는 양수여야 함: {config.batch_size}")
    if not config.data_dir.exists():
        raise ConfigError(f"data_dir 경로 없음: {config.data_dir}")
    ...
```

내부 helper 함수에서는 중복 검증 금지 — 호출자가 보장. Python 의 duck typing 을 신뢰.

## 로깅

### `logging` 모듈 사용

`print` 대신 표준 `logging` 모듈 사용. 연구 노트북에서도 동일.

```python
import logging

logger = logging.getLogger(__name__)

def train_epoch(model, loader, optimizer) -> dict:
    logger.info("epoch 시작", extra={"n_batches": len(loader)})
    for batch_idx, batch in enumerate(loader):
        loss = step(model, batch, optimizer)
        if batch_idx % 100 == 0:
            logger.info(
                "학습 진행",
                extra={"batch_idx": batch_idx, "loss": float(loss)},
            )
    return {"loss": float(loss)}
```

### 로그 레벨

| Level | 사용 시점 | 예시 |
|---|---|---|
| `DEBUG` | 개발/디버깅 상세 — 평소엔 필터링 | 개별 배치 loss, tensor shape 확인 |
| `INFO` | 정상 동작 마일스톤 | epoch 시작/종료, 체크포인트 저장 |
| `WARNING` | 비정상이지만 처리됨 | NaN 1회 발생 후 skip, fallback 사용 |
| `ERROR` | 작업 실패 — 호출자에게 영향 | 데이터 로드 실패, 학습 발산 |
| `CRITICAL` | 복구 불가 — 즉시 중단 | GPU 메모리 부족, 환경 불일치 |

### 구조화된 필드

문자열 보간 대신 `extra` 인자 또는 structured logger (`structlog`) 사용:

```python
# Good
logger.info("eval 완료", extra={
    "split": "test",
    "accuracy": 0.873,
    "loss": 0.42,
    "n_samples": 1024,
})

# Bad — 검색·집계 불가
logger.info(f"eval 완료: test split, acc=0.873, loss=0.42, n=1024")
```

### 메시지 언어

- **로그 message 본문**: 영어 (`epoch start`, `eval complete`) 또는 한국어 일관 — 프로젝트 내에서 한쪽으로 통일. 연구 코드는 한국어 메시지도 허용.
- **사용자 정의 예외 message**: 한국어 허용 (디버깅 시 사용자가 읽는 정보).

## 학습 루프 특수 케이스

### NaN / Inf 감지

```python
loss = model(batch)
if not torch.isfinite(loss):
    logger.error(
        "loss NaN/Inf 감지",
        extra={"step": global_step, "loss": float(loss)},
    )
    raise ModelError(f"loss is not finite at step {global_step}")
```

NaN 은 silently skip 하지 말 것 — 원인 (학습률, 데이터, 초기화) 파악 후 fix.

### 체크포인트 저장 실패

```python
try:
    torch.save(state, ckpt_path)
except (OSError, RuntimeError) as err:
    logger.exception("체크포인트 저장 실패", extra={"path": str(ckpt_path)})
    # 학습은 계속 (다음 epoch 에서 재시도)
```

학습 중단보다 다음 epoch 에서 재시도가 일반적으로 더 가치 있음.

### 결정론 깨짐 감지

```python
def assert_deterministic(seed: int, run_fn) -> None:
    set_seed(seed)
    result_a = run_fn()
    set_seed(seed)
    result_b = run_fn()
    if not torch.allclose(result_a, result_b):
        raise ReproducibilityError(
            f"동일 시드 ({seed}) 에서 결과 불일치 — 결정론 깨짐"
        )
```

CI 또는 디버깅 모드에서 활성화. 매 학습마다는 비용 과다.

## 노트북 특수 케이스

노트북에서는 다음 두 가지를 추가로 신경:

1. **셀 재실행 안전성** — 셀이 두 번 실행돼도 망가지지 않게. 전역 상태 (모델, 옵티마이저) 는 명시적 reset.
2. **에러 후 상태** — 에러로 셀이 중단되면 GPU 메모리 leak 가능. `try/finally` 로 `torch.cuda.empty_cache()` 보장 또는 셀 분리.

```python
# 모델 학습 셀
try:
    train(model, loader, optimizer, epochs=10)
finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```
