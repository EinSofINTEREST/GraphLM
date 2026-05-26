"""Tests for graphlm.neuron.graph_channel — Phase 10 channel-as-node foundations."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from graphlm.neuron.graph_channel import ChannelGraphLinear


def test_shape_after_init():
    lin = ChannelGraphLinear(8, 16)
    assert lin.weight.shape == (16, 8)
    assert lin.adj.shape == (16, 8)
    assert lin.bias.shape == (16,)


def test_full_adj_init_value():
    lin = ChannelGraphLinear(8, 16, adj_init="full")
    assert torch.allclose(lin.adj, torch.ones(16, 8))


def test_uniform_small_adj_init_range():
    """adj_init='uniform_small' → uniform[0.05, 0.15] (Phase 2 sweet spot 패턴)."""
    torch.manual_seed(0)
    lin = ChannelGraphLinear(8, 16, adj_init="uniform_small")
    assert (lin.adj >= 0.05).all() and (lin.adj <= 0.15).all()
    # 분포가 실제로 spread 되어 있는지 (degenerate 거부)
    assert lin.adj.std() > 0.01


@pytest.mark.parametrize("bad_init", ["zero", "zeros"])
def test_zero_init_rejected(bad_init):
    """0-init 옵션 명시적 거부 — feedback_no_zero_init.md 규칙 적용."""
    with pytest.raises(ValueError, match="feedback_no_zero_init"):
        ChannelGraphLinear(8, 16, adj_init=bad_init)  # type: ignore[arg-type]


def test_unknown_adj_init_raises():
    with pytest.raises(ValueError, match="unknown adj_init"):
        ChannelGraphLinear(8, 16, adj_init="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [("in_features", 0), ("in_features", -1), ("out_features", 0), ("out_features", -1)],
)
def test_invalid_features_raises(field, value):
    kwargs = {"in_features": 8, "out_features": 16}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        ChannelGraphLinear(**kwargs)


def test_function_preservation_equivalent_to_standard_linear():
    """adj=full + 같은 W → standard Linear 와 forward 동일 (atol=1e-5)."""
    torch.manual_seed(0)
    in_f, out_f = 8, 16
    cg = ChannelGraphLinear(in_f, out_f, adj_init="full")
    std = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        std.weight.copy_(cg.weight)
        std.bias.copy_(cg.bias)

    x = torch.randn(2, 4, in_f)
    y_cg = cg(x)
    y_std = std(x)
    assert torch.allclose(y_cg, y_std, atol=1e-5), (
        f"function preservation 깨짐: max |diff| = {(y_cg - y_std).abs().max().item()}"
    )


def test_weight_and_adj_both_have_gradient():
    lin = ChannelGraphLinear(8, 16, adj_init="uniform_small")
    x = torch.randn(4, 8)
    out = lin(x)
    out.sum().backward()
    assert lin.weight.grad is not None
    assert lin.adj.grad is not None
    assert (lin.weight.grad.abs().sum() > 0).item()
    assert (lin.adj.grad.abs().sum() > 0).item()


def test_adj_sparsity_initial():
    """초기 adj_init='full' 은 모두 1 이므로 (|adj| < 0.05) sparsity = 0."""
    lin = ChannelGraphLinear(8, 16, adj_init="full")
    assert lin.adj_sparsity(threshold=0.05) == 0.0


def test_adj_sparsity_after_zero_fill():
    lin = ChannelGraphLinear(8, 16, adj_init="full")
    # 일부 entry 를 강제로 0 으로 → sparsity 변화 확인
    with torch.no_grad():
        lin.adj[:8].fill_(0.0)  # 절반 (8/16)
    assert abs(lin.adj_sparsity(threshold=0.05) - 0.5) < 1e-6


def test_sparsify_adj_zeros_below_threshold():
    """uniform_small (0.05~0.15) 에서 threshold=0.10 으로 sparsify → 평균 절반 zeroed."""
    torch.manual_seed(0)
    lin = ChannelGraphLinear(64, 64, adj_init="uniform_small")
    n_total = 64 * 64
    n_zeroed = lin.sparsify_adj(threshold=0.10)
    # uniform[0.05, 0.15] 의 mid = 0.10 이라 약 절반 (~50%) 이 < 0.10
    assert n_zeroed > n_total * 0.3
    assert n_zeroed < n_total * 0.7


def test_sparsify_adj_negative_threshold_raises():
    lin = ChannelGraphLinear(8, 16)
    with pytest.raises(ValueError, match="threshold"):
        lin.sparsify_adj(threshold=-0.1)


def test_freeze_adjacency():
    lin = ChannelGraphLinear(8, 16)
    lin.freeze_adjacency()
    assert not lin.adj.requires_grad
    assert lin.weight.requires_grad
