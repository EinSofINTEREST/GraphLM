"""Tests for HybridTransformerTrainConfig validation (Phase 15)."""

from __future__ import annotations

import pytest

from graphlm.data.tinyshakespeare import CharTokenizer, TinyShakespeareDataset
from graphlm.neuron.hybrid_transformer_demo import HybridTransformerTrainConfig


def _dummy_dataset() -> TinyShakespeareDataset:
    text = "abcdefghij" * 100
    tok = CharTokenizer(text)
    return TinyShakespeareDataset(text, tok)


def _base_kwargs() -> dict:
    ds = _dummy_dataset()
    return dict(
        dataset=ds,
        vocab_size=10,
        hidden_dim=16,
        n_heads=4,
        ffn_dim=32,
        n_layers=2,
        group_size=4,
        arch="hybrid_full_full",
        block_size=8,
        batch_size=4,
        lr=1e-3,
        max_steps=100,
    )


def test_default_config_valid():
    """Phase 15 인자 default (prune_at_step=None, prune_fraction=0.0) 가 유효."""
    cfg = HybridTransformerTrainConfig(**_base_kwargs())
    assert cfg.prune_at_step is None
    assert cfg.prune_fraction == 0.0


@pytest.mark.parametrize("frac", [-0.1, 1.1, 2.0])
def test_invalid_prune_fraction_rejected(frac):
    """prune_fraction ∉ [0, 1] 는 __post_init__ 에서 거부 (Copilot #3307536553)."""
    with pytest.raises(ValueError, match=r"prune_fraction must be in \[0, 1\]"):
        HybridTransformerTrainConfig(**_base_kwargs(), prune_fraction=frac)


@pytest.mark.parametrize("step", [0, -1, 101, 9999])
def test_invalid_prune_at_step_rejected(step):
    """prune_at_step ∉ [1, max_steps] 는 거부."""
    with pytest.raises(ValueError, match="prune_at_step must be in"):
        HybridTransformerTrainConfig(**_base_kwargs(), prune_at_step=step)


def test_valid_prune_config_accepted():
    cfg = HybridTransformerTrainConfig(**_base_kwargs(), prune_at_step=50, prune_fraction=0.3)
    assert cfg.prune_at_step == 50
    assert cfg.prune_fraction == 0.3
