"""Numerical metrics helpers — safe perplexity 등.

노트북에서 직접 정의하던 helper 들을 ``src/graphlm/`` 로 이전 — 노트북은 analysis flow
+ 시각화에만 집중 (CodeRabbit #3304306120, project rule).
"""

from __future__ import annotations

import math


def safe_perplexity(loss: float, cap: float = 20.0) -> float:
    """``exp(loss)`` overflow 방지.

    학습 초반 큰 loss 또는 발산 시 ``math.exp(loss)`` 는 ``OverflowError`` 발생.
    cap=20 → max perplexity ≈ 4.85e8 (충분히 큰 ceiling, 발산 식별 가능).

    Args:
        loss: cross-entropy loss (음수 또는 양수, 무한 가능).
        cap: exp 적용 전 loss 의 상한 (default 20.0).

    Returns:
        ``math.exp(min(loss, cap))`` — 항상 유한 양수.

    Raises:
        ValueError: ``cap`` 이 음수면 (의도적 down-clip 으로 underflow 가능).

    See Also:
        - gemini #3303153101 rationale: log-domain loss → exp-domain perplexity 변환 시
          overflow 회피.
    """
    if cap < 0:
        raise ValueError(f"cap must be non-negative, got {cap}")
    if math.isnan(loss):
        return math.nan
    return math.exp(min(loss, cap))
