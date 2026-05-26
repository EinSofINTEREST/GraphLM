"""Phase 14 — HybridGraphCausalSelfAttention (qkv + out 을 HybridGraphLinear 로).

Phase 13 은 FFN 만 graph 였고 attention 은 표준 ``nn.Linear``. Phase 14 는 **qkv / out 도
HybridGraphLinear** 로 교체하여 block 전체가 graph 가 되는 단계.

설계:
- ``qkv``: hidden_dim → 3·hidden_dim (rectangular — identity outer 미지원)
- ``out``: hidden_dim → hidden_dim (square — identity 이론상 가능하나 통일성 위해 미사용)
- sdpa (scaled dot-product attention) 은 그대로 standard
- function preservation: adj_outer=full + adj_inner=full + 같은 W → standard CausalSelfAttention forward 동치

0-init 거부 + magnitude rule 은 underlying ``HybridGraphLinear`` 가 상속.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.neuron.graph_hybrid import AdjInnerInit, AdjOuterInit, HybridGraphLinear


class HybridGraphCausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with HybridGraphLinear qkv + out.

    Args:
        hidden_dim: Transformer hidden size (n_heads 의 배수).
        n_heads: number of attention heads.
        group_size: HybridGraphLinear 의 block size. hidden_dim 과 3·hidden_dim 모두 배수여야 함.
        adj_outer_init: ``"full"`` / ``"uniform_around_one"`` (identity 는 qkv rectangular 라 미지원).
        adj_inner_init: ``"full"`` / ``"uniform_around_one"``.
        dropout: attention dropout (sdpa 의 dropout_p).

    Forward:
        x: ``(B, T, hidden_dim)`` → y: same shape
        function preserving when both adj = full and W = same as standard nn.Linear init.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        group_size: int,
        *,
        adj_outer_init: AdjOuterInit = "full",
        adj_inner_init: AdjInnerInit = "full",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}")
        if adj_outer_init == "identity":
            raise ValueError(
                "HybridGraphCausalSelfAttention 은 adj_outer_init='identity' 미지원 — "
                "qkv 는 hidden_dim → 3·hidden_dim 으로 rectangular 라 정방 identity 정의 불가. "
                "'full' 또는 'uniform_around_one' 사용."
            )
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = HybridGraphLinear(
            hidden_dim,
            3 * hidden_dim,
            group_size=group_size,
            adj_outer_init=adj_outer_init,
            adj_inner_init=adj_inner_init,
            bias=False,
        )
        self.out = HybridGraphLinear(
            hidden_dim,
            hidden_dim,
            group_size=group_size,
            adj_outer_init=adj_outer_init,
            adj_inner_init=adj_inner_init,
            bias=False,
        )
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)
