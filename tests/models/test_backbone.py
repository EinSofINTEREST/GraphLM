"""Backbone unit tests — shape, dynamic depth, parameter accounting."""

from __future__ import annotations

import torch

from graphlm.models.backbone import BackboneConfig, GrowingDecoder


def test_forward_shape():
    cfg = BackboneConfig(vocab_size=100, hidden_dim=64, n_heads=4, ffn_dim=128, n_init_layers=2)
    model = GrowingDecoder(cfg)
    x = torch.randint(0, 100, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 100)


def test_n_layers_starts_with_init():
    cfg = BackboneConfig(vocab_size=50, hidden_dim=64, n_heads=4, ffn_dim=128, n_init_layers=3)
    model = GrowingDecoder(cfg)
    assert model.n_layers == 3


def test_add_block_increases_n_layers():
    cfg = BackboneConfig(vocab_size=50, hidden_dim=64, n_heads=4, ffn_dim=128, n_init_layers=2)
    model = GrowingDecoder(cfg)
    initial = model.n_layers
    idx = model.add_block(residual_scale=0.0)
    assert idx == initial  # 0-based index of newly added
    assert model.n_layers == initial + 1


def test_add_block_increases_n_params():
    cfg = BackboneConfig(vocab_size=50, hidden_dim=64, n_heads=4, ffn_dim=128, n_init_layers=2)
    model = GrowingDecoder(cfg)
    n_before = model.n_params
    model.add_block(residual_scale=0.0)
    assert model.n_params > n_before


def test_seq_len_exceeds_max_raises():
    cfg = BackboneConfig(
        vocab_size=50, hidden_dim=64, n_heads=4, ffn_dim=128, max_seq_len=8, n_init_layers=2
    )
    model = GrowingDecoder(cfg)
    x = torch.randint(0, 50, (1, 16))
    try:
        model(x)
    except ValueError:
        return
    raise AssertionError("expected ValueError for seq_len > max_seq_len")
