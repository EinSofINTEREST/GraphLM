"""Adaptive triggers — Phase 1: AutoGrow plateau detector.

AutoGrow (Wen et al., AAAI 2020) 의 핵심 idea:
    sliding window 의 validation accuracy / loss 표준편차 σ_W < ε → \"plateau\" → trigger.

본 implementation 은 loss 기준 (값이 낮을수록 좋음). σ < ε 만으로는 \"낮은 정점 plateau\"
와 \"높은 정점 plateau\" 를 구분 못 하므로 cooldown (직전 trigger 후 일정 step 대기) 도 함께
적용 — 연속 trigger 폭주 방지.
"""

from __future__ import annotations

from collections import deque


class PlateauTrigger:
    """Sliding window 의 loss 표준편차로 plateau 감지.

    Usage:
        trigger = PlateauTrigger(window=500, epsilon=0.01, cooldown=500)
        for step, loss in enumerate(train_loop):
            if trigger.update(loss):
                # plateau detected — perform growth
                ...

    Args:
        window: sliding window 크기 (steps 단위).
        epsilon: σ < epsilon 이면 plateau 로 판정.
        cooldown: 직전 trigger 발생 후 다음 trigger 까지 최소 step 간격.
        min_history: trigger 판정 시작에 필요한 최소 history (window 미만 시 false).
    """

    def __init__(
        self,
        *,
        window: int = 500,
        epsilon: float = 0.01,
        cooldown: int = 500,
        min_history: int | None = None,
    ):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.epsilon = epsilon
        self.cooldown = cooldown
        self.min_history = min_history if min_history is not None else window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._last_trigger_step: int | None = None
        self._step: int = 0

    def update(self, loss: float) -> bool:
        """Add a new loss value and return True if plateau detected.

        Incremental O(1) sum + sum_sq update — 큰 window 에서도 매 step overhead 최소.

        Returns:
            True if `update` triggers a grow event at this step, else False.
        """
        v = float(loss)
        # buffer 가 가득 차서 가장 오래된 값이 밀려나면 sum / sum_sq 에서도 제거
        if len(self._buf) == self.window:
            old = self._buf[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._buf.append(v)
        self._sum += v
        self._sum_sq += v * v
        self._step += 1

        if len(self._buf) < self.min_history:
            return False

        # cooldown 검사
        if (
            self._last_trigger_step is not None
            and (self._step - self._last_trigger_step) < self.cooldown
        ):
            return False

        # population variance = E[X²] - (E[X])²
        n = len(self._buf)
        mean = self._sum / n
        var = max(self._sum_sq / n - mean * mean, 0.0)  # floating-point 음수 방지
        sigma = var**0.5

        if sigma < self.epsilon:
            self._last_trigger_step = self._step
            return True
        return False

    @property
    def step(self) -> int:
        return self._step

    @property
    def last_trigger_step(self) -> int | None:
        return self._last_trigger_step
