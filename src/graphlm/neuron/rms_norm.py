"""Phase 13 — RMSNorm (root-mean-square layer normalization).

Modern Transformer (LLaMA / Mistral / Gemma) 의 표준 norm. nn.LayerNorm 대비:

- mean centering 생략 → 연산량 ↓
- bias 파라미터 없음 → 파라미터 수 ↓ (hidden_dim 만큼 절약)
- numerical 안정성 동등 또는 우위 (특히 large hidden_dim)

수식: ``y = x / RMS(x) * weight``, ``RMS(x) = sqrt(mean(x^2) + eps)``

본 paradigm 의 Phase 13 backbone (HybridGraphTransformer) 에서 nn.LayerNorm 대체.
backbone.py 의 기존 LayerNorm 은 Phase 1~12 호환성 위해 그대로 유지.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Args:
        hidden_dim: 정규화 대상 마지막 축의 크기.
        eps: RMS 분모 numerical 안정성 (LLaMA 의 1e-6 동일).

    Shape:
        - input: ``(..., hidden_dim)``
        - output: same as input
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError(f"hidden_dim must be a positive int, got {hidden_dim!r}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.hidden_dim = hidden_dim
        self.eps = eps
        # weight 만 학습 (LayerNorm 의 bias 없음) — function preservation 위해 1.0 으로 시작
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected last dim {self.hidden_dim}, got {x.shape[-1]}")
        # float32 로 cast 해서 RMS 계산 (mixed precision 안전성)
        x_dtype = x.dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x_f * rms).to(x_dtype)
        return y * self.weight

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, eps={self.eps}"
