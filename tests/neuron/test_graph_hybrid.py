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
    """edge_mask=1 (초기) 일 때 forward 가 mask 없이 계산한 값과 정확히 동일 (function preservation).

    Copilot #3307536521 — 이름과 검증 일치: mask=1 의 forward 가 mask 적용 안 한 직접 계산과 같은지 직접 비교.
    """
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 24, group_size=4)
    x = torch.randn(2, 8, 16)
    y = lin(x)

    # mask 없는 forward 직접 계산 (mask=1 이므로 결과 동일해야 함)
    with torch.no_grad():
        x_g = x.reshape(2, 8, lin.n_groups_in, lin.group_size)
        eff_w_no_mask = lin.adj_outer.unsqueeze(-1).unsqueeze(-1) * lin.adj_inner * lin.weight
        y_g = torch.einsum("...gi,Ggik->...Gk", x_g, eff_w_no_mask)
        expected = y_g.reshape(2, 8, lin.out_features) + lin.bias
    assert torch.allclose(y, expected, atol=1e-6), (
        f"mask=1 forward 가 mask 없는 계산과 달라짐, max |diff| = {(y - expected).abs().max().item()}"
    )

    # 추가 sanity: mask 를 모두 0 으로 만들면 출력은 bias 만 남음
    with torch.no_grad():
        lin.edge_mask.zero_()
    y_zero = lin(x)
    bias_expected = lin.bias.expand(2, 8, 24)
    assert torch.allclose(y_zero, bias_expected, atol=1e-6)


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
    """fraction=0.3 → 살아있는 edge 의 정확히 30% prune (topk 기반 deterministic).

    gemini #3307531745 — topk 로 정확 n개 prune 보장되므로 tolerance 불필요.
    """
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    alive_before = lin.n_alive_edges()
    n_pruned = lin.prune_bottom_fraction(0.3)
    expected = int(alive_before * 0.3)
    assert n_pruned == expected, f"target {expected}, got {n_pruned}"


def test_prune_bottom_fraction_tie_breaking_deterministic():
    """모든 magnitude 가 동률일 때도 정확히 n_to_prune 만큼만 prune (no over-prune).

    gemini #3307531740 의 시나리오 — uniform 같은 magnitude → kthvalue 방식은 100% prune 위험.
    """
    lin = HybridGraphLinear(16, 16, group_size=4)
    # 모든 magnitude 가 정확히 같도록 설정
    with torch.no_grad():
        lin.weight.fill_(1.0)
        lin.adj_outer.fill_(1.0)
        lin.adj_inner.fill_(1.0)
    alive_before = lin.n_alive_edges()
    n_pruned = lin.prune_bottom_fraction(0.3)
    expected = int(alive_before * 0.3)
    assert n_pruned == expected
    # over-prune 안 됨 — 살아있는 edge 수가 expected 만큼 줄음
    assert lin.n_alive_edges() == alive_before - expected


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


# ── Phase 16a: edge regrow (DST) ─────────────────────────────


def test_n_pruned_edges_consistent_with_alive():
    lin = HybridGraphLinear(16, 16, group_size=4)
    total = lin.edge_mask.numel()
    assert lin.n_pruned_edges() + lin.n_alive_edges() == total


def test_regrow_random_basic():
    """random regrow 가 정확히 n 개 살리고 alive count 증가."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    alive_before = lin.n_alive_edges()
    n_regrown = lin.regrow_random(50, reset_weight=True)
    assert n_regrown == 50
    assert lin.n_alive_edges() == alive_before + 50


def test_regrow_random_reset_weight_zeros():
    """reset_weight=True → 새로 살린 위치의 weight=0."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    pruned_before = lin.edge_mask == 0
    n = lin.regrow_random(30, reset_weight=True)
    # 새로 살린 위치 = pruned_before AND mask_now>0
    new_alive = pruned_before & (lin.edge_mask > 0)
    assert int(new_alive.sum().item()) == n
    assert torch.all(lin.weight.data[new_alive] == 0.0)


def test_regrow_random_no_reset_preserves_weight():
    """reset_weight=False → weight 값 보존."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    # weight 의 prune 위치 값 기록
    pruned_before = lin.edge_mask == 0
    weight_at_pruned = lin.weight.data[pruned_before].clone()
    lin.regrow_random(30, reset_weight=False)
    # 일부 위치는 다시 alive 됨 — 그 위치의 weight 가 원래 값과 같은지 (변경 없음) 확인
    # 모든 pruned_before 위치의 weight 가 변경 안 됐어야 함
    assert torch.equal(lin.weight.data[pruned_before], weight_at_pruned)


def test_regrow_more_than_pruned_caps_at_pruned():
    """n > n_pruned 이면 모든 pruned 만큼만 regrow."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.3)
    n_pruned = lin.n_pruned_edges()
    n = lin.regrow_random(99999)
    assert n == n_pruned
    assert lin.n_pruned_edges() == 0


def test_regrow_zero_noop():
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    assert lin.regrow_random(0) == 0


def test_regrow_negative_rejected():
    lin = HybridGraphLinear(16, 16, group_size=4)
    with pytest.raises(ValueError, match="n must be >= 0"):
        lin.regrow_random(-1)


def test_regrow_by_score_picks_top_n():
    """scores 기반 regrow 가 정확히 top-n 위치 선택."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    # scores: pruned 위치마다 다른 값 (rank 가능)
    scores = torch.zeros_like(lin.edge_mask)
    pruned_indices = torch.nonzero(lin.edge_mask == 0, as_tuple=False)
    # 첫 10 위치에 가장 큰 score
    for rank, (i0, i1, i2, i3) in enumerate(pruned_indices[:10]):
        scores[i0, i1, i2, i3] = 100.0 - rank
    n = lin.regrow_by_score(5, scores)
    assert n == 5
    # 첫 5 위치만 alive 됐는지
    for i0, i1, i2, i3 in pruned_indices[:5]:
        assert lin.edge_mask[i0, i1, i2, i3] == 1.0
    # 다음 5 위치는 여전히 pruned
    for i0, i1, i2, i3 in pruned_indices[5:10]:
        assert lin.edge_mask[i0, i1, i2, i3] == 0.0


def test_regrow_by_score_wrong_shape_rejected():
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    bad_scores = torch.zeros(10)
    with pytest.raises(ValueError, match="shape"):
        lin.regrow_by_score(3, bad_scores)


def test_regrow_then_forward_gradient_flows():
    """regrow 후 forward 가 새 edge 에서 정상 grad 흐름 (mask=1 효과)."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(
        16,
        16,
        group_size=4,
        adj_outer_init="uniform_around_one",
        adj_inner_init="uniform_around_one",
    )
    lin.prune_bottom_fraction(0.5)
    # regrow 한 위치
    pruned_before = lin.edge_mask == 0
    lin.regrow_random(30, reset_weight=True)
    regrown_positions = pruned_before & (lin.edge_mask > 0)
    # forward + backward — regrown 위치는 weight=0 이지만 mask=1 이라 grad 는 흘러야 함
    x = torch.randn(4, 16)
    lin(x).sum().backward()
    # regrown 위치의 weight grad 가 nonzero (= 살아있음 증명)
    grad_at_regrown = lin.weight.grad[regrown_positions]
    assert (grad_at_regrown.abs().sum() > 0).item(), (
        "regrow 후 새 edge 의 weight grad 가 모두 0 — gradient 가 안 흐름"
    )


def test_constant_sparsity_prune_then_regrow():
    """prune N + regrow N → sparsity 변경 없음 (DST constant sparsity)."""
    torch.manual_seed(0)
    lin = HybridGraphLinear(16, 16, group_size=4)
    lin.prune_bottom_fraction(0.5)
    sparsity_after_prune = lin.effective_sparsity()
    # 동량 regrow
    n_prune = lin.n_pruned_edges()
    target_swap = int(n_prune * 0.2)  # 20% swap
    lin.prune_bottom_fraction(0.2)  # alive 의 20% 추가 prune
    lin.regrow_random(target_swap)
    # 정확히 같지는 않으나 근방 — prune+regrow 동량이면 sparsity 일정
    # 더 정확히: 새 prune 수 == regrow 수면 sparsity 동일
    sparsity_after_dst = lin.effective_sparsity()
    # 두 sparsity 차이는 prune/regrow 의 정확성에 따라 +/-
    assert abs(sparsity_after_dst - sparsity_after_prune) < 0.1


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
