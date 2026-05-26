"""Tests for graphlm.utils.metrics — safe_perplexity etc."""

from __future__ import annotations

import math

import pytest

from graphlm.utils import safe_perplexity


def test_normal_loss():
    """일반적인 char-LM loss 값 (~2.0) 이 정상 perplexity 반환."""
    assert safe_perplexity(2.0) == pytest.approx(math.exp(2.0))


def test_caps_at_default_20():
    """loss=30 → cap=20 적용 → exp(20)."""
    assert safe_perplexity(30.0) == pytest.approx(math.exp(20.0))


def test_below_cap_unchanged():
    """loss < cap 일 때 cap 영향 없음."""
    assert safe_perplexity(5.0, cap=20.0) == pytest.approx(math.exp(5.0))


def test_negative_loss():
    """음수 loss 도 정상 처리 (분류 confidence 높을 때)."""
    assert safe_perplexity(-1.0) == pytest.approx(math.exp(-1.0))


def test_custom_cap():
    """cap 인자로 ceiling 조정 가능."""
    assert safe_perplexity(15.0, cap=10.0) == pytest.approx(math.exp(10.0))


def test_nan_input_returns_nan():
    """NaN loss → NaN perplexity (silent overflow 회피)."""
    assert math.isnan(safe_perplexity(math.nan))


def test_inf_input_caps():
    """+inf loss → cap 적용 → 유한 값."""
    assert safe_perplexity(math.inf) == pytest.approx(math.exp(20.0))


def test_neg_cap_rejected():
    """음수 cap 거부 — exp(음수 large) underflow 위험."""
    with pytest.raises(ValueError, match="non-negative"):
        safe_perplexity(2.0, cap=-1.0)
