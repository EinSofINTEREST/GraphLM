"""Tests for graphlm.neuron.rms_norm — Phase 13 RMSNorm."""

from __future__ import annotations

import pytest
import torch

from graphlm.neuron.rms_norm import RMSNorm


def test_shape_preserved():
    norm = RMSNorm(16)
    x = torch.randn(2, 8, 16)
    assert norm(x).shape == x.shape


def test_init_weight_is_ones():
    norm = RMSNorm(16)
    assert torch.allclose(norm.weight, torch.ones(16))


def test_rms_normalizes_to_unit_rms():
    """초기 weight=1 에서 출력의 RMS 가 1 에 매우 가까워야 함."""
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 5.0  # arbitrary scale
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    # eps 때문에 정확히 1 은 아니지만 매우 근사
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_weight_is_learnable():
    norm = RMSNorm(16)
    x = torch.randn(2, 16)
    out = norm(x)
    out.sum().backward()
    assert norm.weight.grad is not None
    assert (norm.weight.grad.abs().sum() > 0).item()


def test_zero_dim_raises():
    with pytest.raises(ValueError, match="positive int"):
        RMSNorm(0)


def test_negative_eps_raises():
    with pytest.raises(ValueError, match="eps must be > 0"):
        RMSNorm(16, eps=-1e-6)


def test_wrong_last_dim_raises():
    norm = RMSNorm(16)
    with pytest.raises(ValueError, match="expected last dim 16"):
        norm(torch.randn(2, 8))


def test_scaling_via_weight():
    """weight=2.0 로 setting → 출력도 2배 scale."""
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(4, 16)
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-3)


def test_non_floating_input_raises():
    """int 등 비-floating 입력은 silent cast 회피 위해 차단 (Copilot #3303168589)."""
    norm = RMSNorm(16)
    x_int = torch.zeros(4, 16, dtype=torch.long)
    with pytest.raises(TypeError, match="floating-point"):
        norm(x_int)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_roundtrip_mixed_precision(dtype):
    """출력 dtype 이 입력 dtype 과 일치 (gemini #3303153077). residual connection 안전."""
    norm = RMSNorm(32)
    x = torch.randn(4, 32, dtype=dtype)
    y = norm(x)
    assert y.dtype == dtype, f"expected {dtype}, got {y.dtype}"
