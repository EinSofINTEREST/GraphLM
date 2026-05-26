"""Tests for graphlm.neuron.graph_attention — Phase 14 graph attention."""

from __future__ import annotations

import pytest
import torch

from graphlm.neuron.backbone import CausalSelfAttention
from graphlm.neuron.graph_attention import HybridGraphCausalSelfAttention
from graphlm.neuron.graph_hybrid import HybridGraphLinear


def test_shape():
    attn = HybridGraphCausalSelfAttention(hidden_dim=32, n_heads=4, group_size=8)
    x = torch.randn(2, 16, 32)
    assert attn(x).shape == (2, 16, 32)


def test_qkv_out_are_hybrid_graph_linear():
    attn = HybridGraphCausalSelfAttention(hidden_dim=32, n_heads=4, group_size=8)
    assert isinstance(attn.qkv, HybridGraphLinear)
    assert isinstance(attn.out, HybridGraphLinear)
    # qkv: 32 → 96, out: 32 → 32
    assert attn.qkv.in_features == 32
    assert attn.qkv.out_features == 96
    assert attn.out.in_features == 32
    assert attn.out.out_features == 32


def test_identity_outer_rejected():
    """qkv 가 rectangular 라 identity outer 미지원."""
    with pytest.raises(ValueError, match="rectangular"):
        HybridGraphCausalSelfAttention(
            hidden_dim=32, n_heads=4, group_size=8, adj_outer_init="identity"
        )


def test_heads_not_divisible_raises():
    with pytest.raises(ValueError, match="not divisible by n_heads"):
        HybridGraphCausalSelfAttention(hidden_dim=33, n_heads=4, group_size=11)


def test_function_preservation_full_full_matches_standard():
    """adj=full/full + 같은 W → standard CausalSelfAttention 와 forward 동치 (atol=1e-5)."""
    torch.manual_seed(0)
    hidden, n_heads, k = 32, 4, 8
    hg_attn = HybridGraphCausalSelfAttention(hidden, n_heads, group_size=k)
    std_attn = CausalSelfAttention(hidden, n_heads, dropout=0.0)

    _copy_hybrid_to_plain(hg_attn.qkv, std_attn.qkv)
    _copy_hybrid_to_plain(hg_attn.out, std_attn.out)

    hg_attn.eval()
    std_attn.eval()
    x = torch.randn(2, 8, hidden)
    with torch.no_grad():
        y_hg = hg_attn(x)
        y_std = std_attn(x)
    assert torch.allclose(y_hg, y_std, atol=1e-5), (
        f"attention forward 차이: max |diff| = {(y_hg - y_std).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "outer,inner",
    [
        ("full", "full"),
        ("full", "uniform_around_one"),
        ("uniform_around_one", "uniform_around_one"),
    ],
)
def test_gradient_flows_all_params(outer, inner):
    attn = HybridGraphCausalSelfAttention(
        hidden_dim=16,
        n_heads=4,
        group_size=4,
        adj_outer_init=outer,
        adj_inner_init=inner,
    )
    x = torch.randn(2, 4, 16, requires_grad=False)
    attn(x).sum().backward()
    null_grad = [n for n, p in attn.named_parameters() if p.grad is None]
    assert not null_grad, f"grad 없는 파라미터: {null_grad}"


@pytest.mark.parametrize("bad", ["zero", "zeros"])
def test_zero_init_outer_rejected(bad):
    """0-init outer 거부 — underlying HybridGraphLinear 로부터 상속."""
    with pytest.raises(ValueError, match="vanishing"):
        HybridGraphCausalSelfAttention(
            hidden_dim=32,
            n_heads=4,
            group_size=8,
            adj_outer_init=bad,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", ["zero", "zeros"])
def test_zero_init_inner_rejected(bad):
    """0-init inner 거부."""
    with pytest.raises(ValueError, match="vanishing"):
        HybridGraphCausalSelfAttention(
            hidden_dim=32,
            n_heads=4,
            group_size=8,
            adj_inner_init=bad,  # type: ignore[arg-type]
        )


# ── helpers ──


def _copy_hybrid_to_plain(hg, plain):
    """HybridGraphLinear 의 block weight → nn.Linear 표준 weight 형식으로 복사."""
    in_f, out_f = hg.in_features, hg.out_features
    k = hg.group_size
    W_std = torch.zeros(out_f, in_f)
    for go in range(hg.n_groups_out):
        for gi in range(hg.n_groups_in):
            W_std[go * k : (go + 1) * k, gi * k : (gi + 1) * k] = hg.weight[go, gi].T
    with torch.no_grad():
        plain.weight.copy_(W_std)
