"""Tests for graphlm.neuron.hybrid_transformer — Phase 13 Transformer integration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from graphlm.neuron.hybrid_transformer import (
    HybridGraphFFN,
    HybridGraphTransformerBlock,
    PlainFFN,
    PlainTransformerBlock,
    make_block,
)

# ── HybridGraphFFN ───────────────────────────────────────────


def test_ffn_shape():
    ffn = HybridGraphFFN(hidden_dim=32, ffn_dim=64, group_size=8)
    x = torch.randn(2, 16, 32)
    assert ffn(x).shape == (2, 16, 32)


def test_ffn_identity_outer_rejected():
    """FFN 은 rectangular 라 identity outer 미지원."""
    with pytest.raises(ValueError, match="rectangular"):
        HybridGraphFFN(hidden_dim=32, ffn_dim=64, group_size=8, adj_outer_init="identity")


def test_ffn_function_preservation_full_full():
    """adj_outer=full + adj_inner=full + 같은 weight → standard PlainFFN 동일."""
    torch.manual_seed(0)
    hidden, ffn_d, k = 16, 32, 4
    hg_ffn = HybridGraphFFN(hidden, ffn_d, group_size=k)
    plain_ffn = PlainFFN(hidden, ffn_d)

    # hg_ffn 의 block weight 를 standard nn.Linear weight 로 변환해서 plain 에 복사
    _copy_hybrid_to_plain(hg_ffn.fc1, plain_ffn.fc1)
    _copy_hybrid_to_plain(hg_ffn.fc2, plain_ffn.fc2)

    x = torch.randn(2, 8, hidden)
    y_hg = hg_ffn(x)
    y_plain = plain_ffn(x)
    assert torch.allclose(y_hg, y_plain, atol=1e-5), (
        f"function preservation 깨짐: max |diff| = {(y_hg - y_plain).abs().max().item()}"
    )


def test_ffn_all_params_have_gradient():
    ffn = HybridGraphFFN(
        16,
        32,
        group_size=4,
        adj_outer_init="uniform_around_one",
        adj_inner_init="uniform_around_one",
    )
    x = torch.randn(2, 16)
    ffn(x).sum().backward()
    for layer in [ffn.fc1, ffn.fc2]:
        for attr in ["weight", "adj_outer", "adj_inner"]:
            grad = getattr(layer, attr).grad
            assert grad is not None, f"{layer}.{attr}.grad is None"
            assert (grad.abs().sum() > 0).item(), f"{layer}.{attr}.grad all zero"


# ── HybridGraphTransformerBlock ──────────────────────────────


def test_block_shape():
    block = HybridGraphTransformerBlock(hidden_dim=32, n_heads=4, ffn_dim=64, group_size=8)
    x = torch.randn(2, 16, 32)
    assert block(x).shape == (2, 16, 32)


def test_block_function_preservation_against_plain():
    """hybrid block (adj=full/full) + plain block 의 동일 weight 로 forward 동일."""
    torch.manual_seed(0)
    hidden, n_heads, ffn_d, k = 16, 4, 32, 4
    hybrid = HybridGraphTransformerBlock(hidden, n_heads, ffn_d, group_size=k)
    plain = PlainTransformerBlock(hidden, n_heads, ffn_d)

    # rms / attn 은 standard module 이므로 state_dict copy 가능
    plain.rms1.load_state_dict(hybrid.rms1.state_dict())
    plain.rms2.load_state_dict(hybrid.rms2.state_dict())
    plain.attn.load_state_dict(hybrid.attn.state_dict())
    # FFN 만 block → standard 변환
    _copy_hybrid_to_plain(hybrid.ffn.fc1, plain.ffn.fc1)
    _copy_hybrid_to_plain(hybrid.ffn.fc2, plain.ffn.fc2)

    x = torch.randn(2, 8, hidden)
    y_hybrid = hybrid(x)
    y_plain = plain(x)
    assert torch.allclose(y_hybrid, y_plain, atol=1e-5), (
        f"block forward 차이: max |diff| = {(y_hybrid - y_plain).abs().max().item()}"
    )


def test_block_gradient_flows_all_params():
    block = HybridGraphTransformerBlock(
        hidden_dim=16,
        n_heads=4,
        ffn_dim=32,
        group_size=4,
        adj_outer_init="uniform_around_one",
        adj_inner_init="uniform_around_one",
    )
    x = torch.randn(2, 4, 16)
    block(x).sum().backward()
    null_grad = [n for n, p in block.named_parameters() if p.grad is None]
    assert not null_grad, f"grad 없는 파라미터: {null_grad}"


# ── make_block dispatch ──────────────────────────────────────


@pytest.mark.parametrize(
    "arch",
    ["plain", "hybrid_full_full", "hybrid_full_around_one", "hybrid_around_one_around_one"],
)
def test_make_block_all_archs_forward(arch):
    block = make_block(arch, hidden_dim=16, n_heads=4, ffn_dim=32, group_size=4)
    x = torch.randn(2, 8, 16)
    assert block(x).shape == (2, 8, 16)


def test_make_block_unknown_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        make_block("bogus", 16, 4, 32, 4)  # type: ignore[arg-type]


def test_make_block_plain_uses_nn_linear():
    block = make_block("plain", 16, 4, 32, 4)
    assert isinstance(block, PlainTransformerBlock)
    assert isinstance(block.ffn.fc1, nn.Linear)


def test_make_block_hybrid_uses_hybrid_ffn():
    block = make_block("hybrid_full_full", 16, 4, 32, 4)
    assert isinstance(block, HybridGraphTransformerBlock)
    assert isinstance(block.ffn, HybridGraphFFN)


# ── dropout consistency (Copilot #3303168649 / #3303168686) ──


def test_hybrid_ffn_applies_dropout_when_training():
    """HybridGraphFFN 도 fc2 뒤 dropout — backbone.FFN 패턴 일관."""
    ffn = HybridGraphFFN(16, 32, group_size=4, dropout=0.5)
    ffn.train()
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    # dropout 활성 시 stochastic — eval 모드와 다른 output 보장
    y_train = ffn(x)
    ffn.eval()
    y_eval = ffn(x)
    assert not torch.allclose(y_train, y_eval), "dropout 이 train 모드에서 적용되지 않음"


def test_plain_ffn_applies_dropout_when_training():
    """PlainFFN 도 dropout 인자 적용 — Hybrid 와 API 일관."""
    ffn = PlainFFN(16, 32, dropout=0.5)
    ffn.train()
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y_train = ffn(x)
    ffn.eval()
    y_eval = ffn(x)
    assert not torch.allclose(y_train, y_eval), "dropout 이 train 모드에서 적용되지 않음"


@pytest.mark.parametrize(
    "block_cls,kwargs",
    [
        (
            PlainTransformerBlock,
            {"hidden_dim": 16, "n_heads": 4, "ffn_dim": 32, "dropout": 0.5},
        ),
        (
            HybridGraphTransformerBlock,
            {"hidden_dim": 16, "n_heads": 4, "ffn_dim": 32, "group_size": 4, "dropout": 0.5},
        ),
    ],
)
def test_block_dropout_propagates_to_ffn(block_cls, kwargs):
    """block 의 dropout 인자가 attention 만이 아니라 FFN 까지 전달."""
    block = block_cls(**kwargs)
    assert block.ffn.dropout.p == 0.5, "FFN 의 dropout p 가 block dropout 과 불일치"


# ── helpers ──────────────────────────────────────────────────


def _copy_hybrid_to_plain(hg, plain):
    """HybridGraphLinear 의 block weight → nn.Linear 표준 weight 형식으로 복사.

    Phase 12 test_function_preservation_full_full_equivalent_to_linear 와 동일 로직.
    """
    in_f, out_f = hg.in_features, hg.out_features
    k = hg.group_size
    W_std = torch.zeros(out_f, in_f)
    for go in range(hg.n_groups_out):
        for gi in range(hg.n_groups_in):
            W_std[go * k : (go + 1) * k, gi * k : (gi + 1) * k] = hg.weight[go, gi].T
    with torch.no_grad():
        plain.weight.copy_(W_std)
        if plain.bias is not None and hg.bias is not None:
            plain.bias.copy_(hg.bias)
