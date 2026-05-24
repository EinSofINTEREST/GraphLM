"""PlateauTrigger unit tests — plateau 감지 + cooldown + min_history."""

from __future__ import annotations

import pytest

from graphlm.training.triggers import PlateauTrigger


def test_min_history_blocks_early_trigger():
    trigger = PlateauTrigger(window=10, epsilon=0.01, cooldown=0, min_history=10)
    # 9 step 동안은 절대 fire X
    for _ in range(9):
        assert trigger.update(0.5) is False
    # 10번째 부터 fire 가능
    assert trigger.update(0.5) is True


def test_plateau_detected_when_constant_loss():
    trigger = PlateauTrigger(window=5, epsilon=0.01, cooldown=0, min_history=5)
    fired = [trigger.update(0.5) for _ in range(10)]
    # 첫 5 회 채운 후 6번째 (index 5) 부터 fire 가능
    assert any(fired)


def test_no_plateau_when_loss_varies():
    trigger = PlateauTrigger(window=10, epsilon=0.001, cooldown=0, min_history=10)
    # 매 step 마다 0.5 변동 — σ 가 epsilon 보다 훨씬 큼
    fired = [trigger.update(i * 0.5) for i in range(20)]
    assert not any(fired)


def test_cooldown_blocks_consecutive_triggers():
    trigger = PlateauTrigger(window=5, epsilon=0.01, cooldown=10, min_history=5)
    # 5 step 채워서 첫 trigger
    for _ in range(5):
        trigger.update(0.5)
    # 다음 update 가 fire — 정확히 어느 시점인지 확인
    fired_steps = []
    for _ in range(20):
        if trigger.update(0.5):
            fired_steps.append(trigger.step)
    # cooldown=10 이므로 두 fire 사이 최소 10 step
    if len(fired_steps) >= 2:
        assert fired_steps[1] - fired_steps[0] >= 10


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        PlateauTrigger(window=1, epsilon=0.01)
