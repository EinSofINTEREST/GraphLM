"""Function-level dynamic architecture — `neuron` paradigm.

Attention module 단위의 dynamic add + learnable α. 기존 `graphlm.models` (block-level)
와 병렬 / 별도 line. Csordás et al. 2021 (Transformer attention head 가 가장 modular)
의 발견을 기반으로 head-level granularity 검증.
"""

from graphlm.neuron.backbone import NeuronBlock, NeuronConfig, NeuronGrowingDecoder
from graphlm.neuron.growth import add_attn_function_preserving

__all__ = [
    "NeuronBlock",
    "NeuronConfig",
    "NeuronGrowingDecoder",
    "add_attn_function_preserving",
]
