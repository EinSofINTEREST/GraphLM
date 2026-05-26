"""Phase 13 — HybridGraphLinear 의 Transformer FFN 통합 + RMSNorm pre-norm block.

본 paradigm 의 graph 표현 (Phase 12 HybridGraphLinear) 을 **실제 Transformer 아키텍처** 안의
FFN 위치에 도입. Phase 8~12 는 MLP-LM baseline 이었고, Phase 13 부터 standard pre-norm
Transformer block 위에서 검증.

Scope (Phase 13):
- FFN 의 fc1 (hidden → ffn) + fc2 (ffn → hidden) 만 ``HybridGraphLinear``
- attention (qkv / out) 은 표준 ``nn.Linear`` 유지 (Phase 14 검토)
- norm 은 ``RMSNorm`` (modern Transformer 표준)

function preservation:
- ``HybridGraphFFN`` 의 adj_outer=full + adj_inner=full + 동일 weight 초기화 → standard FFN 동치
- ``HybridGraphTransformerBlock`` 의 FFN 만 hybrid, attention/norm 은 표준 → standard pre-norm block 과 동치
- 검증은 tests/neuron/test_hybrid_transformer.py 참조
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.neuron.backbone import CausalSelfAttention
from graphlm.neuron.graph_hybrid import AdjInnerInit, AdjOuterInit, HybridGraphLinear
from graphlm.neuron.rms_norm import RMSNorm


class HybridGraphFFN(nn.Module):
    """FFN with two ``HybridGraphLinear`` layers + GELU.

    Args:
        hidden_dim: in/out 차원 (Transformer hidden size).
        ffn_dim: 중간 확장 차원 (보통 4·hidden_dim).
        group_size: ``HybridGraphLinear`` 의 block size. hidden_dim / ffn_dim 모두 배수여야 함.
        adj_outer_init: outer adj 초기화 (``"full"`` / ``"uniform_around_one"``).
            ``"identity"`` 는 FFN 이 정의상 rectangular (hidden ≠ ffn) 이라 지원하지 않음.
        adj_inner_init: inner adj 초기화 (``"full"`` / ``"uniform_around_one"``).

    Forward:
        ``y = fc2(GELU(fc1(x)))`` — function preserving when both adj = full.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        group_size: int,
        *,
        adj_outer_init: AdjOuterInit = "full",
        adj_inner_init: AdjInnerInit = "full",
    ):
        super().__init__()
        if adj_outer_init == "identity":
            raise ValueError(
                "HybridGraphFFN 은 adj_outer_init='identity' 미지원 — "
                "FFN 은 hidden_dim → ffn_dim → hidden_dim 으로 rectangular 라 정방 identity 정의 불가. "
                "'full' 또는 'uniform_around_one' 사용."
            )
        self.fc1 = HybridGraphLinear(
            hidden_dim,
            ffn_dim,
            group_size=group_size,
            adj_outer_init=adj_outer_init,
            adj_inner_init=adj_inner_init,
            bias=False,
        )
        self.fc2 = HybridGraphLinear(
            ffn_dim,
            hidden_dim,
            group_size=group_size,
            adj_outer_init=adj_outer_init,
            adj_inner_init=adj_inner_init,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class HybridGraphTransformerBlock(nn.Module):
    """Pre-norm Transformer block — RMSNorm + CausalSelfAttention + RMSNorm + HybridGraphFFN.

    Forward:
        ``x = x + attn(rms1(x))``
        ``x = x + ffn(rms2(x))``
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        group_size: int,
        *,
        adj_outer_init: AdjOuterInit = "full",
        adj_inner_init: AdjInnerInit = "full",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rms1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout=dropout)
        self.rms2 = RMSNorm(hidden_dim)
        self.ffn = HybridGraphFFN(
            hidden_dim,
            ffn_dim,
            group_size=group_size,
            adj_outer_init=adj_outer_init,
            adj_inner_init=adj_inner_init,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rms1(x))
        return x + self.ffn(self.rms2(x))


Arch = Literal[
    "plain",
    "hybrid_full_full",
    "hybrid_full_around_one",
    "hybrid_around_one_around_one",
]


class PlainFFN(nn.Module):
    """Standard 2-layer FFN (no bias) — Phase 13 ``"plain"`` baseline."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PlainTransformerBlock(nn.Module):
    """Pre-norm Transformer block with standard FFN — Phase 13 ``"plain"`` baseline.

    HybridGraphTransformerBlock 과 동일한 norm/attention/residual 구조 — FFN 만 표준 nn.Linear.
    공정 비교 위해 RMSNorm 동일 사용.
    """

    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.rms1 = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout=dropout)
        self.rms2 = RMSNorm(hidden_dim)
        self.ffn = PlainFFN(hidden_dim, ffn_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rms1(x))
        return x + self.ffn(self.rms2(x))


def make_block(
    arch: Arch,
    hidden_dim: int,
    n_heads: int,
    ffn_dim: int,
    group_size: int,
    dropout: float = 0.0,
) -> nn.Module:
    """4 가지 arch 중 하나로 Phase 13 Transformer block 생성."""
    if arch == "plain":
        return PlainTransformerBlock(hidden_dim, n_heads, ffn_dim, dropout=dropout)
    if arch == "hybrid_full_full":
        return HybridGraphTransformerBlock(
            hidden_dim,
            n_heads,
            ffn_dim,
            group_size=group_size,
            adj_outer_init="full",
            adj_inner_init="full",
            dropout=dropout,
        )
    if arch == "hybrid_full_around_one":
        return HybridGraphTransformerBlock(
            hidden_dim,
            n_heads,
            ffn_dim,
            group_size=group_size,
            adj_outer_init="full",
            adj_inner_init="uniform_around_one",
            dropout=dropout,
        )
    if arch == "hybrid_around_one_around_one":
        return HybridGraphTransformerBlock(
            hidden_dim,
            n_heads,
            ffn_dim,
            group_size=group_size,
            adj_outer_init="uniform_around_one",
            adj_inner_init="uniform_around_one",
            dropout=dropout,
        )
    raise ValueError(f"unknown arch: {arch}")
