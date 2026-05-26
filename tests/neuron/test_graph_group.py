"""Tests for graphlm.neuron.graph_group — Phase 9 group-as-node foundations."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from graphlm.neuron.graph_group import GroupGraphLinear


def test_shape_after_init():
    lin = GroupGraphLinear(16, 24, group_size=4)
    assert lin.n_groups_in == 4
    assert lin.n_groups_out == 6
    assert lin.weight.shape == (6, 4, 4, 4)
    assert lin.adj.shape == (6, 4)
    assert lin.bias.shape == (24,)


def test_forward_shape():
    lin = GroupGraphLinear(16, 24, group_size=4)
    x = torch.randn(2, 8, 16)  # (B, T, in_features)
    y = lin(x)
    assert y.shape == (2, 8, 24)


def test_in_features_not_divisible_raises():
    with pytest.raises(ValueError, match="in_features.*divisible"):
        GroupGraphLinear(17, 24, group_size=4)


def test_out_features_not_divisible_raises():
    with pytest.raises(ValueError, match="out_features.*divisible"):
        GroupGraphLinear(16, 25, group_size=4)


def test_full_adj_init_value():
    lin = GroupGraphLinear(16, 24, group_size=4, adj_init="full")
    assert torch.allclose(lin.adj, torch.ones(6, 4))


def test_identity_adj_init_value():
    lin = GroupGraphLinear(16, 16, group_size=4, adj_init="identity")
    assert torch.allclose(lin.adj, torch.eye(4))


def test_invalid_adj_init_raises():
    with pytest.raises(ValueError, match="unknown adj_init"):
        GroupGraphLinear(16, 16, group_size=4, adj_init="bogus")  # type: ignore[arg-type]


def test_function_preservation_equivalent_to_standard_linear():
    """adj=full + Linear 와 같은 W 로 init 하면 forward 결과 동일.

    수학적으로: GroupGraphLinear (adj=1 모든 곳) = standard Linear (블록을 모두 모은 W).
    """
    torch.manual_seed(0)
    in_f, out_f, k = 16, 24, 4
    gg = GroupGraphLinear(in_f, out_f, group_size=k, adj_init="full")
    # gg.weight: shape (G_out=6, G_in=4, k=4, k=4) — 각 block 이 standard W 의 block tile
    # standard W 와 같은 effective W 를 만들려면: W_std[go*k:(go+1)*k, gi*k:(gi+1)*k] = gg.weight[go, gi].T
    # (linear.weight 는 (out, in) shape, x @ W^T 형식)
    G_out, G_in = gg.n_groups_out, gg.n_groups_in
    W_std = torch.zeros(out_f, in_f)
    for go in range(G_out):
        for gi in range(G_in):
            W_std[go * k : (go + 1) * k, gi * k : (gi + 1) * k] = gg.weight[go, gi].T
    std = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        std.weight.copy_(W_std)
        std.bias.copy_(gg.bias)

    x = torch.randn(2, 8, in_f)
    y_gg = gg(x)
    y_std = std(x)
    assert torch.allclose(y_gg, y_std, atol=1e-5), (
        f"function preservation 깨짐: max |diff| = {(y_gg - y_std).abs().max().item()}"
    )


def test_identity_init_zero_off_diagonal_contribution():
    """adj=identity init 시 off-diagonal group 의 routing = 0 → block-diagonal forward."""
    torch.manual_seed(0)
    in_f, out_f, k = 16, 16, 4  # n_groups_in = n_groups_out = 4
    lin = GroupGraphLinear(in_f, out_f, group_size=k, adj_init="identity")
    x = torch.randn(2, in_f)
    y = lin(x)
    # 각 출력 그룹 [go*k:(go+1)*k] 는 *동일* 입력 그룹 [go*k:(go+1)*k] 만의 함수여야 함
    # → x 의 다른 group entry 변경이 영향 없음 확인
    x_perturbed = x.clone()
    # group 1 (indices 4~7) 만 변경
    x_perturbed[:, 4:8] += 100
    y_perturbed = lin(x_perturbed)
    # group 0 (indices 0~3) 출력은 안 변해야
    assert torch.allclose(y[:, 0:4], y_perturbed[:, 0:4], atol=1e-5)
    # group 1 출력은 변해야
    assert not torch.allclose(y[:, 4:8], y_perturbed[:, 4:8], atol=1e-3)


def test_weight_and_adj_both_have_gradient():
    lin = GroupGraphLinear(16, 24, group_size=4)
    x = torch.randn(2, 16)
    out = lin(x)
    out.sum().backward()
    assert lin.weight.grad is not None
    assert lin.adj.grad is not None
    assert (lin.weight.grad.abs().sum() > 0).item()
    assert (lin.adj.grad.abs().sum() > 0).item()


def test_freeze_adjacency():
    lin = GroupGraphLinear(16, 24, group_size=4)
    lin.freeze_adjacency()
    assert not lin.adj.requires_grad
    # weight 는 여전히 학습 가능
    assert lin.weight.requires_grad


def test_sparsify_adjacency_zeros_below_threshold():
    lin = GroupGraphLinear(16, 24, group_size=4, adj_init="full")
    # all-ones (=1.0) → threshold=1.5 면 모두 0 으로 강제
    n_zeroed = lin.sparsify_adjacency(threshold=1.5)
    assert n_zeroed == 6 * 4  # 모든 entry
    assert torch.allclose(lin.adj, torch.zeros_like(lin.adj))


def test_sparsify_adjacency_negative_threshold_raises():
    lin = GroupGraphLinear(16, 24, group_size=4)
    with pytest.raises(ValueError, match="threshold"):
        lin.sparsify_adjacency(threshold=-0.1)
