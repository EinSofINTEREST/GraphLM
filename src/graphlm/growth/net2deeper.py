"""Net2DeeperNet — function-preserving depth growth (Net2Net, ICLR 2016).

본 프로젝트 phase 1 에서의 단순 형태:
- 새 DecoderBlock 을 `alpha=0` 으로 끝에 append → forward output 동일 (function preservation).
- 학습이 진행되며 alpha 가 0 에서 의미 있는 값으로 학습됨.

Net2Net 의 \"identity initialization\" 을 modern Pre-LN Transformer 의 residual scale 방식으로 적용.
원논문은 ReLU 의 strict identity 였으나 본 구현은 \"residual scale 0 = skip\" 이라는 더
일반적이고 modern (GradMax / MSG 와 호환) 형태.
"""

from __future__ import annotations

from graphlm.models.backbone import GrowingDecoder


def add_layer_function_preserving(model: GrowingDecoder) -> int:
    """Append a new DecoderBlock to the model with alpha=0 (function-preserving).

    Args:
        model: GrowingDecoder instance.

    Returns:
        Index of the newly added block (0-based, equals model.n_layers - 1 after the call).

    Invariant:
        For all input_ids x: model.forward(x) before call == model.forward(x) after call.
        Verified by `assert_function_preserving` helper.
    """
    return model.add_block(residual_scale=0.0)
