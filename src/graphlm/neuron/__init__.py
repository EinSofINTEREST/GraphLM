"""Function-level dynamic architecture — `neuron` paradigm.

Attention module 단위의 dynamic add + learnable α. 기존 `graphlm.models` (block-level)
와 병렬 / 별도 line. Csordás et al. 2021 (Transformer attention head 가 가장 modular)
의 발견을 기반으로 head-level granularity 검증.
"""

from graphlm.neuron.backbone import (
    NeuronBlock,
    NeuronConfig,
    NeuronGrowingDecoder,
    SinusoidalAlpha,
)
from graphlm.neuron.graph_channel import ChannelGraphLinear
from graphlm.neuron.graph_group import GroupGraphLinear
from graphlm.neuron.graph_attention import HybridGraphCausalSelfAttention
from graphlm.neuron.graph_hybrid import HybridGraphLinear
from graphlm.neuron.growable import GrowableEmbedding, GrowableLayerNorm, GrowableLinear
from graphlm.neuron.growth import add_attn_function_preserving, add_attn_smooth_start
from graphlm.neuron.hybrid_transformer import (
    FullGraphTransformerBlock,
    HybridGraphFFN,
    HybridGraphTransformerBlock,
    PlainTransformerBlock,
    make_block,
    make_full_block,
)
from graphlm.neuron.rms_norm import RMSNorm

__all__ = [
    "ChannelGraphLinear",
    "FullGraphTransformerBlock",
    "GroupGraphLinear",
    "GrowableEmbedding",
    "GrowableLayerNorm",
    "GrowableLinear",
    "HybridGraphCausalSelfAttention",
    "HybridGraphFFN",
    "HybridGraphLinear",
    "HybridGraphTransformerBlock",
    "NeuronBlock",
    "NeuronConfig",
    "NeuronGrowingDecoder",
    "PlainTransformerBlock",
    "RMSNorm",
    "SinusoidalAlpha",
    "add_attn_function_preserving",
    "add_attn_smooth_start",
    "make_block",
    "make_full_block",
]
