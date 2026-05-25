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


def add_attn_smooth_start(
    model: NeuronGrowingDecoder, block_idx: int, alpha_init: float = 0.01
) -> int:
    """주어진 block 에 새 attention 추가 — α=alpha_init (nonzero) 으로 init.

    Phase 1 의 dead block 회피를 위한 옵션 D. α=0 의 function preservation 을 약간
    양보 (추가 직후 작은 forward 변화 = 작은 loss spike) 하는 대신 추가 attn 의 weight
    가 처음부터 의미 있는 gradient flow 받음 → α 0 갇힘 회피 가능성 확보.

    Args:
        model: 대상 NeuronGrowingDecoder.
        block_idx: 어느 block 에 추가할지.
        alpha_init: 신규 attn 의 초기 α (default 0.01). 0.0 이면
            `add_attn_function_preserving` 과 동등.

    Returns:
        새 attention 의 within-block index.

    Raises:
        ValueError: alpha_init < 0.
    """
    if alpha_init < 0:
        raise ValueError(f"alpha_init must be >= 0, got {alpha_init}")
    return model.add_attn(block_idx, residual_scale=alpha_init)
