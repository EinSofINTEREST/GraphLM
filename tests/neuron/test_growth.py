"""Tests for graphlm.neuron.growth — function preservation invariant."""

from __future__ import annotations

import pytest
import torch

from graphlm.neuron.backbone import NeuronConfig, NeuronGrowingDecoder
from graphlm.neuron.growth import add_attn_function_preserving


@pytest.fixture
def small_cfg():
    return NeuronConfig(
        vocab_size=32,
        hidden_dim=64,
        n_heads=2,
        ffn_dim=128,
        max_seq_len=16,
        n_layers=2,
        n_init_attn=1,
    )


def test_add_attn_preserves_function(small_cfg):
    """새 attention 추가 직후 forward 가 변하지 않아야 (alpha=0 init)."""
    torch.manual_seed(0)
    model = NeuronGrowingDecoder(small_cfg)
    model.eval()
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        add_attn_function_preserving(model, block_idx=0)
        out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6), (
        f"function preservation violated: max diff = {(out_before - out_after).abs().max().item()}"
    )


def test_add_attn_multiple_blocks_preserve(small_cfg):
    """여러 block 에 차례로 추가해도 forward 변하지 않아야."""
    torch.manual_seed(0)
    model = NeuronGrowingDecoder(small_cfg)
    model.eval()
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        for block_idx in range(small_cfg.n_layers):
            add_attn_function_preserving(model, block_idx)
        out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_add_attn_alpha_nonzero_changes_output(small_cfg):
    """alpha=nonzero 로 추가하면 forward 가 변해야 (sanity)."""
    torch.manual_seed(0)
    model = NeuronGrowingDecoder(small_cfg)
    model.eval()
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        model.add_attn(0, residual_scale=1.0)
        out_after = model(x)
    # alpha=1 + random init → output 변경 기대
    assert not torch.allclose(out_before, out_after, atol=1e-4)


def test_add_attn_smooth_start_default_changes_output(small_cfg):
    """add_attn_smooth_start (alpha_init=0.01 default) 는 작은 forward 변화 발생."""
    from graphlm.neuron.growth import add_attn_smooth_start

    torch.manual_seed(0)
    model = NeuronGrowingDecoder(small_cfg)
    model.eval()
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        add_attn_smooth_start(model, block_idx=0)
        out_after = model(x)
    # alpha=0.01 → 작은 변화. atol=1e-6 으로는 통과 안 됨 (변화 존재), atol=0.5 으론 통과 (작은 변화)
    assert not torch.allclose(out_before, out_after, atol=1e-6)
    assert torch.allclose(out_before, out_after, atol=0.5)


def test_add_attn_smooth_start_alpha_zero_preserves(small_cfg):
    """alpha_init=0.0 이면 function preservation (= add_attn_function_preserving 동등)."""
    from graphlm.neuron.growth import add_attn_smooth_start

    torch.manual_seed(0)
    model = NeuronGrowingDecoder(small_cfg)
    model.eval()
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out_before = model(x)
        add_attn_smooth_start(model, block_idx=0, alpha_init=0.0)
        out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_add_attn_smooth_start_negative_raises(small_cfg):
    from graphlm.neuron.growth import add_attn_smooth_start

    model = NeuronGrowingDecoder(small_cfg)
    with pytest.raises(ValueError, match="alpha_init"):
        add_attn_smooth_start(model, block_idx=0, alpha_init=-0.01)
