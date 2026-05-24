"""Net2DeeperNet 의 function preservation 검증 — 본 PR 의 가장 중요한 invariant."""

from __future__ import annotations

import pytest
import torch

from graphlm.growth.function_preservation import (
    FunctionPreservationError,
    assert_function_preserving,
)
from graphlm.growth.net2deeper import add_layer_function_preserving
from graphlm.models.backbone import BackboneConfig, GrowingDecoder


def _make_model(seed: int = 42) -> GrowingDecoder:
    torch.manual_seed(seed)
    cfg = BackboneConfig(vocab_size=100, hidden_dim=64, n_heads=4, ffn_dim=128, n_init_layers=2)
    return GrowingDecoder(cfg).eval()


def test_add_layer_preserves_function():
    """가장 중요한 테스트 — 새 block 추가가 output 을 바꾸지 않아야 함."""
    model = _make_model()
    x = torch.randint(0, 100, (2, 16))
    assert_function_preserving(
        grow_op=lambda: add_layer_function_preserving(model),
        forward_fn=model.forward,
        sample_input=x,
    )
    # n_layers 증가도 확인
    assert model.n_layers == 3


def test_multiple_add_layer_calls_all_preserve():
    """연속 호출도 각 호출마다 preservation 유지."""
    model = _make_model()
    x = torch.randint(0, 100, (1, 32))
    for _ in range(3):
        assert_function_preserving(
            grow_op=lambda: add_layer_function_preserving(model),
            forward_fn=model.forward,
            sample_input=x,
        )
    assert model.n_layers == 5


def test_function_preservation_detects_breakage():
    """일부러 alpha != 0 으로 add 하면 preservation 위반 → 에러 발생 확인."""
    model = _make_model()
    x = torch.randint(0, 100, (1, 16))
    with pytest.raises(FunctionPreservationError):
        assert_function_preserving(
            grow_op=lambda: model.add_block(residual_scale=1.0),  # alpha=1 → 변화
            forward_fn=model.forward,
            sample_input=x,
        )
