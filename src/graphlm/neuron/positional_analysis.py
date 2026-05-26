"""Phase 6: positional α (SinusoidalAlpha) 분석 헬퍼.

학습된 SinusoidalAlpha 의 amplitude/bias/log_freq 를 (T, hidden_dim) 매트릭스로 평가하기 위한
순수 함수. 노트북에서 직접 정의하지 않고 import — `.claude/rules/06-code-style.md` 의 "노트북
셀 안에서 로직 정의 금지" 규약 준수.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def eval_positional_alpha_from_state(state: dict[str, Any], seq_len: int) -> np.ndarray:
    """노트북에 저장된 SinusoidalAlpha state dict (tensor 또는 ndarray) 로부터 (T, hidden_dim) 평가.

    각 entry 는 ``amplitude`` / ``bias`` / ``log_freq`` 키. tensor 면 ``.numpy()`` 변환 후 평가.
    """

    def _to_np(v: Any) -> np.ndarray:
        return v.numpy() if hasattr(v, "numpy") else np.asarray(v)

    return eval_positional_alpha(
        _to_np(state["amplitude"]),
        _to_np(state["bias"]),
        _to_np(state["log_freq"]),
        seq_len,
    )


def eval_positional_alpha(
    amplitude: np.ndarray, bias: np.ndarray, log_freq: np.ndarray, seq_len: int
) -> np.ndarray:
    """학습된 SinusoidalAlpha 파라미터를 (T, hidden_dim) ndarray 로 평가.

    α(t)[c] = amplitude[c] * sin(t * exp(log_freq[c])) + bias[c]

    Args:
        amplitude: shape ``(hidden_dim,)`` — 채널별 진폭.
        bias: shape ``(hidden_dim,)`` — 채널별 baseline.
        log_freq: shape ``(hidden_dim,)`` — 채널별 log(주파수).
        seq_len: 평가할 시퀀스 길이.

    Returns:
        shape ``(seq_len, hidden_dim)`` ndarray.
    """
    freq = np.exp(log_freq)
    t = np.arange(seq_len)[:, None]
    return amplitude[None, :] * np.sin(t * freq[None, :]) + bias[None, :]
