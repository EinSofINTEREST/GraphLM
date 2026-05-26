"""Tests for graphlm.neuron.graph_hybrid — Phase 12 hierarchical hybrid foundations."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from graphlm.neuron.graph_hybrid import HybridGraphLinear


def test_shape_after_init():
    lin = HybridGraphLinear(16, 24, group_size=4)
    assert lin.n_groups_in == 4
    assert lin.n_groups_out == 6
    # weight: (G_out, G_in, k, k)
    assert lin.weight.shape == (6, 4, 4, 4)
    # adj_outer: (G_out, G_in)
    assert lin.adj_outer.shape == (6, 4)
    # adj_inner: (G_out, G_in, k, k)
    assert lin.adj_inner.shape == (6, 4, 4, 4)
    assert lin.bias.shape == (24,)


def test_forward_shape():
    lin = HybridGraphLinear(16, 24, group_size=4)
    x = torch.randn(2, 8, 16)
    y = lin(x)
    assert y.shape == (2, 8, 24)


@pytest.mark.parametrize(
    "outer,inner",
    [("full", "full"), ("identity", "full"), ("uniform_around_one", "uniform_around_one")],
)
def test_adj_init_combinations_valid(outer, inner):
    lin = HybridGraphLinear(16, 16, group_size=4, adj_outer_init=outer, adj_inner_init=inner)
    assert lin.adj_outer.shape == (4, 4)
    assert lin.adj_inner.shape == (4, 4, 4, 4)


def test_identity_adj_outer_requires_square():
    """adj_outer_init='identity' 는 정방 (n_groups_out==n_groups_in) 만 허용."""
    with pytest.raises(ValueError, match="requires square"):
        HybridGraphLinear(16, 24, group_size=4, adj_outer_init="identity")


def test_function_preservation_full_full_equivalent_to_linear():
    """adj_outer=full + adj_inner=full + 같은 W → standard Linear forward 동일 (atol=1e-5)."""
    torch.manual_seed(0)
    in_f, out_f, k = 16, 24, 4
    hg = HybridGraphLinear(in_f, out_f, group_size=k, adj_outer_init="full", adj_inner_init="full")

    # standard Linear W = blocks 모은 형태 (block (go, gi, k, k) 의 transpose)
    G_out, G_in = hg.n_groups_out, hg.n_groups_in
    W_std = torch.zeros(out_f, in_f)
    for go in range(G_out):
        for gi in range(G_in):
            # hg.weight[go, gi] shape (k, k) — block matmul 에서 x[gi] @ W[go, gi] 이므로
            # standard Linear (y = W @ x) 의 block 은 W_std[go*k:(go+1)*k, gi*k:(gi+1)*k] = hg.weight[go, gi].T
            W_std[go * k : (go + 1) * k, gi * k : (gi + 1) * k] = hg.weight[go, gi].T

    std = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        std.weight.copy_(W_std)
        std.bias.copy_(hg.bias)

    x = torch.randn(2, 8, in_f)
    y_hg = hg(x)
    y_std = std(x)
    assert torch.allclose(y_hg, y_std, atol=1e-5), (
        f"function preservation 깨짐: max |diff| = {(y_hg - y_std).abs().max().item()}"
    )


@pytest.mark.parametrize("bad", ["zero", "zeros"])
def test_zero_init_outer_rejected(bad):
    """adj_outer_init='zero' 거부 — 0-init 금지 규칙."""
    with pytest.raises(ValueError, match="vanishing"):
        HybridGraphLinear(16, 16, group_size=4, adj_outer_init=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["zero", "zeros"])
def test_zero_init_inner_rejected(bad):
    """adj_inner_init='zero' 거부 — 0-init 금지 규칙."""
    with pytest.raises(ValueError, match="vanishing"):
        HybridGraphLinear(16, 16, group_size=4, adj_inner_init=bad)  # type: ignore[arg-type]


def test_unknown_adj_outer_raises():
    with pytest.raises(ValueError, match="unknown adj_outer_init"):
        HybridGraphLinear(16, 16, group_size=4, adj_outer_init="bogus")  # type: ignore[arg-type]


def test_unknown_adj_inner_raises():
    with pytest.raises(ValueError, match="unknown adj_inner_init"):
        HybridGraphLinear(16, 16, group_size=4, adj_inner_init="bogus")  # type: ignore[arg-type]


def test_all_three_params_have_gradient():
    """weight, adj_outer, adj_inner 모두 grad 흐름."""
    lin = HybridGraphLinear(
        16,
        24,
        group_size=4,
        adj_outer_init="uniform_around_one",
        adj_inner_init="uniform_around_one",
    )
    x = torch.randn(2, 16)
    out = lin(x)
    out.sum().backward()
    for attr in ["weight", "adj_outer", "adj_inner"]:
        grad = getattr(lin, attr).grad
        assert grad is not None, f"{attr}.grad is None"
        assert (grad.abs().sum() > 0).item(), f"{attr}.grad all zero"


def test_in_features_not_divisible_raises():
    with pytest.raises(ValueError, match="in_features.*divisible"):
        HybridGraphLinear(17, 24, group_size=4)


def test_invalid_features_raises():
    with pytest.raises(ValueError, match="positive int"):
        HybridGraphLinear(0, 16, group_size=4)


def test_sparsity_metrics_initial():
    """초기 (full/full) 은 모두 1 → sparsity 0."""
    lin = HybridGraphLinear(16, 16, group_size=4)
    assert lin.adj_outer_sparsity(0.05) == 0.0
    assert lin.adj_inner_sparsity(0.05) == 0.0


def test_freeze_helpers():
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.freeze_adj_outer()
    assert not lin.adj_outer.requires_grad
    assert lin.adj_inner.requires_grad
    assert lin.weight.requires_grad
    lin.freeze_adj_inner()
    assert not lin.adj_inner.requires_grad


# ── Phase 15: edge prune ────────────────────────────────────────


def test_edge_mask_initial_all_ones():
    """초기 edge_mask 는 전부 1 (no prune)."""
    lin = HybridGraphLinear(16, 16, group_size=4)
    assert lin.edge_mask.shape == (4, 4, 4, 4)
    assert torch.all(lin.edge_mask == 1.0)
    assert lin.effective_sparsity() == 0.0
    assert lin.n_alive_edges() == 16 * 16  # 4·4·4·4


def test_forward_with_initial_mask_unchanged():
    """edge_mask=1 일 때 forward 는 Phase 14 와 동일 (function preservation)."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 24, group_size=4)
    x = torch.randn(2, 8, 16)
    y = lin(x)
    # mask 다 0 으로 만들면 출력도 bias 만 남아야 함 (sanity check)
    with torch.no_grad():
        lin.edge_mask.zero_()
    y_zero = lin(x)
    expected = lin.bias.expand(2, 8, 24)
    assert torch.allclose(y_zero, expected, atol=1e-6), (
        f"mask=0 시 출력은 bias 만 남아야 함, max |diff| = {(y_zero - expected).abs().max().item()}"
    )
    # 원래 mask 복구해서 y 와 다르다는 것도 확인
    assert not torch.allclose(y, y_zero)


def test_prune_by_magnitude_basic():
    """threshold 이하 edge 가 mask=0 으로 prune."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    initial_mag = lin.effective_edge_magnitude()
    # threshold = 50 percentile 로 잡으면 절반 정도 prune
    threshold = float(initial_mag.median().item())
    n_pruned = lin.prune_by_magnitude(threshold)
    assert n_pruned > 0
    sparsity = lin.effective_sparsity()
    assert 0.4 < sparsity < 0.6, f"median threshold 면 ~50% sparsity, got {sparsity}"


def test_prune_idempotent_below_threshold():
    """같은 threshold 로 두 번 호출 시 두 번째는 0 신규 prune."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    threshold = float(lin.effective_edge_magnitude().median().item())
    n1 = lin.prune_by_magnitude(threshold)
    n2 = lin.prune_by_magnitude(threshold)
    assert n1 > 0
    assert n2 == 0, f"두 번째 prune 은 신규 0 이어야 함, got {n2}"


def test_pruned_edges_do_not_resurrect_via_gradient():
    """pruned edge 의 weight/adj gradient 가 정확히 0 — optimizer 가 살릴 수 없음."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(
        16,
        16,
        group_size=4,
        adj_outer_init="uniform_around_one",
        adj_inner_init="uniform_around_one",
    )
    threshold = float(lin.effective_edge_magnitude().median().item())
    lin.prune_by_magnitude(threshold)
    dead_positions = lin.edge_mask == 0

    x = torch.randn(2, 16)
    lin(x).sum().backward()
    # weight gradient: pruned 위치는 0
    assert torch.all(lin.weight.grad[dead_positions] == 0), "weight grad at pruned == 0"
    # adj_inner gradient: pruned 위치는 0
    assert torch.all(lin.adj_inner.grad[dead_positions] == 0), "adj_inner grad at pruned == 0"


def test_prune_bottom_fraction():
    """fraction=0.3 → 살아있는 edge 의 ~30% 신규 prune."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    alive_before = lin.n_alive_edges()
    n_pruned = lin.prune_bottom_fraction(0.3)
    expected = int(alive_before * 0.3)
    # tie-breaking 으로 약간 초과 가능 → ±10 허용
    assert abs(n_pruned - expected) <= 10, f"target {expected}, got {n_pruned}"


def test_prune_bottom_fraction_zero_fraction_noop():
    lin = HybridGraphLinear(16, 16, group_size=4)
    assert lin.prune_bottom_fraction(0.0) == 0


def test_prune_negative_threshold_rejected():
    lin = HybridGraphLinear(16, 16, group_size=4)
    with pytest.raises(ValueError, match="must be >= 0"):
        lin.prune_by_magnitude(-0.1)


def test_prune_invalid_fraction_rejected():
    lin = HybridGraphLinear(16, 16, group_size=4)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        lin.prune_bottom_fraction(1.5)


def test_edge_mask_in_state_dict():
    """edge_mask 가 state_dict 에 포함되어 save/load 보존."""
    lin1 = HybridGraphLinear(16, 16, group_size=4)
    lin1.prune_by_magnitude(float(lin1.effective_edge_magnitude().median().item()))
    sparsity_before = lin1.effective_sparsity()
    assert sparsity_before > 0

    lin2 = HybridGraphLinear(16, 16, group_size=4)
    lin2.load_state_dict(lin1.state_dict())
    assert lin2.effective_sparsity() == sparsity_before
    assert torch.all(lin2.edge_mask == lin1.edge_mask)
