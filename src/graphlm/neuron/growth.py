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
    """Add a new attention module with a nonzero ``alpha_init`` for dead-block avoidance.

    주어진 block 에 새 attention 추가 — α=alpha_init (nonzero) 으로 init. Phase 1 의
    dead block 회피를 위한 옵션 D. α=0 의 function preservation 을 약간 양보 (추가 직후
    작은 forward 변화 = 작은 loss spike) 하는 대신 추가 attn 의 weight 가 처음부터
    의미 있는 gradient flow 받음 → α 0 갇힘 회피 가능성 확보.

    α shape 은 ``model.cfg.alpha_per_channel`` 에 의해 결정 — scalar (Phase 1~3) 또는
    per-channel vector ∈ ℝ^{hidden_dim} (Phase 4+). 어느 경우든 ``alpha_init`` (float) 는
    모든 채널에 동일하게 적용 (uniform init).

    Args:
        model: target NeuronGrowingDecoder.
        block_idx: target block index (0-based).
        alpha_init: initial α for the new attention (default 0.01).
            0.0 falls back to function-preserving behavior identical to
            ``add_attn_function_preserving``.

    Returns:
        Within-block index of the new attention module.

    Raises:
        ValueError: when ``alpha_init`` is negative.
    """
    if alpha_init < 0:
        raise ValueError(f"alpha_init must be >= 0, got {alpha_init}")
    return model.add_attn(block_idx, residual_scale=alpha_init)
