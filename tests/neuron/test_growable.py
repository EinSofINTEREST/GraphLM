"""Tests for graphlm.neuron.growable — Phase 8 structural axis foundations."""

from __future__ import annotations

import pytest
import torch

from graphlm.neuron.growable import (
    GrowableEmbedding,
    GrowableLayerNorm,
    GrowableLinear,
)

# ---------- GrowableLinear ---------- #


def test_growable_linear_shape_after_expand_out():
    lin = GrowableLinear(8, 16)
    lin.expand_out(4)
    assert lin.weight.shape == (20, 8)
    assert lin.bias.shape == (20,)
    assert lin.out_features == 20


def test_growable_linear_shape_after_expand_in():
    lin = GrowableLinear(8, 16)
    lin.expand_in(3)
    assert lin.weight.shape == (16, 11)
    assert lin.in_features == 11


def test_growable_linear_expand_out_zero_init_preserves_function_for_old_dims():
    """zero-init expand_out 직후: 기존 16 dim 의 forward output 은 변하지 않아야.

    새 4 dim 은 0 으로 채워져 새로 등장 (downstream 의 expand_in 이 zero-init 으로 받으면 전체
    forward 불변).
    """
    torch.manual_seed(0)
    lin = GrowableLinear(8, 16)
    x = torch.randn(2, 8)
    out_before = lin(x).clone()
    lin.expand_out(4, init="zero")
    out_after = lin(x)
    assert torch.allclose(out_before, out_after[..., :16], atol=1e-6)
    # 새 4 dim 의 출력 = 0 (zero weight + zero bias)
    assert torch.allclose(out_after[..., 16:], torch.zeros_like(out_after[..., 16:]))


def test_growable_linear_expand_in_zero_init_preserves_function():
    """zero-init expand_in 직후: 기존 입력 dim 만으로의 forward 는 변하지 않아야.

    새 in dim 의 column 이 0 이므로, 새 input dim 값에 무관하게 출력 동일.
    """
    torch.manual_seed(0)
    lin = GrowableLinear(8, 16)
    x_old = torch.randn(2, 8)
    out_before = lin(x_old).clone()
    lin.expand_in(3, init="zero")
    # 확장된 입력: 기존 8 dim + 새 3 dim (임의 값) → 출력은 기존과 동일해야
    x_new = torch.cat([x_old, torch.randn(2, 3)], dim=1)
    out_after = lin(x_new)
    assert torch.allclose(out_before, out_after, atol=1e-6)


# ---------- AdamW state 보존 ---------- #


def test_growable_linear_expand_out_preserves_adamw_state():
    """expand_out + optimizer → AdamW 의 m/v 가 확장되고 기존 dim 의 값 보존, 새 dim = 0."""
    torch.manual_seed(0)
    lin = GrowableLinear(8, 16)
    optim = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    # 한 번 step 해서 state 채우기
    x = torch.randn(4, 8)
    out = lin(x)
    out.sum().backward()
    optim.step()
    optim.zero_grad()
    # state snapshot — 기존 weight 의 m, v
    old_weight = lin.weight
    m_before = optim.state[old_weight]["exp_avg"].clone()
    v_before = optim.state[old_weight]["exp_avg_sq"].clone()
    step_before = optim.state[old_weight]["step"]
    assert m_before.shape == (16, 8) and v_before.shape == (16, 8)

    lin.expand_out(4, optimizer=optim, init="zero")
    # 새 param 으로 교체됨
    new_weight = lin.weight
    assert new_weight.shape == (20, 8)
    assert new_weight is not old_weight
    # optimizer state 확장 + 기존 보존
    new_state = optim.state[new_weight]
    m_after = new_state["exp_avg"]
    v_after = new_state["exp_avg_sq"]
    assert m_after.shape == (20, 8) and v_after.shape == (20, 8)
    # 기존 16 dim 보존
    assert torch.allclose(m_after[:16], m_before)
    assert torch.allclose(v_after[:16], v_before)
    # 새 4 dim = 0
    assert torch.allclose(m_after[16:], torch.zeros_like(m_after[16:]))
    assert torch.allclose(v_after[16:], torch.zeros_like(v_after[16:]))
    # step counter 보존
    assert new_state["step"] == step_before


def test_growable_linear_expand_preserves_grad_when_present():
    """backward() 후 step() 전에 expand 호출 시 기존 grad 보존 + 새 dim grad=0 (gemini #3300391305)."""
    torch.manual_seed(0)
    lin = GrowableLinear(8, 16)
    x = torch.randn(4, 8)
    out = lin(x)
    out.sum().backward()
    # step() 전에 expand — grad 가 살아있는 상태
    old_grad = lin.weight.grad.clone()
    assert old_grad is not None and old_grad.shape == (16, 8)

    lin.expand_out(4, init="zero")
    new_grad = lin.weight.grad
    assert new_grad is not None and new_grad.shape == (20, 8)
    # 기존 16 dim 의 grad 보존
    assert torch.allclose(new_grad[:16], old_grad)
    # 새 4 dim 의 grad = 0
    assert torch.allclose(new_grad[16:], torch.zeros_like(new_grad[16:]))


def test_growable_linear_expand_repeated_stability():
    """반복 expansion 후에도 forward / backward / step 정상 동작."""
    torch.manual_seed(0)
    lin = GrowableLinear(8, 16)
    optim = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    for _ in range(3):
        x = torch.randn(4, 8)
        out = lin(x)
        loss = out.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
        lin.expand_out(2, optimizer=optim, init="zero")
    # 최종 shape: 16 + 2*3 = 22
    assert lin.out_features == 22
    # 마지막으로 정상 step
    x = torch.randn(4, 8)
    out = lin(x)
    out.sum().backward()
    optim.step()


# ---------- GrowableLayerNorm ---------- #


def test_growable_layernorm_identity_preserving_expand():
    """expand 직후: 기존 dim 의 LN 출력 동일, 새 dim 의 weight=1/bias=0 (identity-style)."""
    torch.manual_seed(0)
    ln = GrowableLayerNorm(16)
    x = torch.randn(2, 4, 16)
    _ = ln(x)
    ln.expand(4)
    assert ln.weight.shape == (20,) and ln.bias.shape == (20,)
    # 새 4 dim weight = 1, bias = 0
    assert torch.allclose(ln.weight[16:], torch.ones(4))
    assert torch.allclose(ln.bias[16:], torch.zeros(4))
    # 새 입력: 기존 + zero (downstream upstream expand_out 의 zero-init 효과 시뮬레이션)
    x_new = torch.cat([x, torch.zeros(2, 4, 4)], dim=-1)
    out_after = ln(x_new)
    # LN 의 mean/var 가 새 dim 0 포함으로 약간 달라지지만, expansion 의 의미는 "새 dim 도
    # downstream 에서 무시되도록 identity 유지" 확인 — weight/bias init 만 검증
    assert out_after.shape == (2, 4, 20)


# ---------- GrowableEmbedding ---------- #


def test_growable_embedding_expand_dim_preserves_old_dims():
    """expand_dim 직후: 기존 embedding_dim 의 출력 동일, 새 dim = 0 (zero init)."""
    torch.manual_seed(0)
    emb = GrowableEmbedding(10, 8)
    idx = torch.tensor([0, 3, 5, 9])
    out_before = emb(idx).clone()
    emb.expand_dim(4, init="zero")
    assert emb.weight.shape == (10, 12)
    out_after = emb(idx)
    assert torch.allclose(out_before, out_after[..., :8], atol=1e-6)
    assert torch.allclose(out_after[..., 8:], torch.zeros_like(out_after[..., 8:]))


# ---------- 메타: invalid init mode ---------- #


def test_growable_linear_invalid_init_mode_raises():
    lin = GrowableLinear(8, 16)
    with pytest.raises(ValueError, match="unknown init mode"):
        lin.expand_out(2, init="bogus")  # type: ignore[arg-type]
