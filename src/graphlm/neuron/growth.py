"""Function-preserving attention module addition for NeuronGrowingDecoder.

기존 `graphlm.growth.net2deeper` 가 block-level 의 add_layer_function_preserving
을 제공한다면, 본 module 은 attention module-level 의 add_attn_function_preserving 을
제공한다. 둘 다 α=0 init 으로 function preservation 보장.
"""

from __future__ import annotations

from graphlm.neuron.backbone import NeuronGrowingDecoder


def add_attn_function_preserving(model: NeuronGrowingDecoder, block_idx: int) -> int:
    """주어진 block 에 새 attention module 을 function-preserving 하게 추가.

    α=0 으로 init 되므로 추가 직후 forward 는 변화 없음 (mathematically identity).

    Args:
        model: 대상 NeuronGrowingDecoder.
        block_idx: 어느 block 에 추가할지 (0-based).

    Returns:
        새 attention 의 within-block index.
    """
    return model.add_attn(block_idx, residual_scale=0.0)
