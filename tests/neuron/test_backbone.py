"""Tests for graphlm.neuron.backbone — NeuronGrowingDecoder."""

from __future__ import annotations

import pytest
import torch

from graphlm.neuron.backbone import NeuronConfig, NeuronGrowingDecoder


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


def test_forward_shape(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, small_cfg.vocab_size)


def test_n_attn_per_block_initial(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    assert model.n_attn_per_block == [1, 1]


def test_add_attn_increments_count(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    model.add_attn(0)
    model.add_attn(0)
    model.add_attn(1)
    assert model.n_attn_per_block == [3, 2]


def test_add_attn_invalid_block_idx_raises(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    with pytest.raises(IndexError, match="block_idx"):
        model.add_attn(99)


def test_add_attn_alpha_zero_default(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    model.add_attn(0)
    # 새 attention 은 alpha=0.0 으로 init (function preservation)
    new_alpha = model.blocks[0].attn_alphas[-1].item()
    assert new_alpha == 0.0


def test_n_params_increases_after_add_attn(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    before = model.n_params
    model.add_attn(0)
    after = model.n_params
    # 새 LN (2*hidden) + new qkv (hidden * 3*hidden) + new out (hidden * hidden) + alpha (1)
    expected_delta = (
        2 * small_cfg.hidden_dim  # LN weight + bias
        + small_cfg.hidden_dim * 3 * small_cfg.hidden_dim  # qkv
        + small_cfg.hidden_dim * small_cfg.hidden_dim  # out
        + 1  # alpha
    )
    assert after - before == expected_delta


def test_forward_after_add_attn(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    model.add_attn(0)
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, small_cfg.vocab_size)


def test_max_seq_len_exceeded_raises(small_cfg):
    model = NeuronGrowingDecoder(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (1, small_cfg.max_seq_len + 1))
    with pytest.raises(ValueError, match="seq_len"):
        model(x)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("n_layers", 0, "n_layers"),
        ("n_init_attn", 0, "n_init_attn"),
        ("hidden_dim", 0, "hidden_dim"),
        ("n_heads", 0, "n_heads"),
        ("vocab_size", 0, "vocab_size"),
        ("max_seq_len", 0, "max_seq_len"),
    ],
)
def test_config_validation_rejects_zero(field, value, match):
    """NeuronConfig.__post_init__ 가 잘못된 값을 명시적 ValueError 로 거부."""
    kwargs = {
        "vocab_size": 32,
        "hidden_dim": 64,
        "n_heads": 2,
        "ffn_dim": 128,
        "max_seq_len": 16,
        "n_layers": 2,
        "n_init_attn": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        NeuronConfig(**kwargs)


def test_config_validation_hidden_dim_not_divisible():
    with pytest.raises(ValueError, match="divisible"):
        NeuronConfig(vocab_size=32, hidden_dim=65, n_heads=2)
