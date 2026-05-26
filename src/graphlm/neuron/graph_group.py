"""Phase 9 — group-as-node graph hidden layer foundations.

채널을 ``group_size`` 단위로 묶어 graph node 로 다룬다. dense ``nn.Linear`` 대신 그룹 간
adjacency 가 결정하는 block-sparse routing.

```
hidden_dim H 채널을 k 개씩 묶어 G = H/k 그룹 → G 개 node
within-group: dense (k × k) block weight
between-group: 학습된 adjacency adj[go, gi] 가 routing scalar
forward: y[go] = Σ_gi adj[go, gi] · (x[gi] @ W[go, gi])
```

이 구조의 의미:
- node = k 채널 묶음 (graph 의 vertex)
- edge = group 간 block routing (k×k block weight × adj scalar)
- adjacency = 1 모두 + identity-ish weight init → standard Linear 와 forward 동치 (function preservation)
- adjacency 의 일부 entry 를 0 으로 sparsify → block-sparse routing emerges

참고 ML 패턴: MoE (Switch Transformer) 의 expert routing, multi-head attention 의 head grouping,
DeepSeek-V3 의 block-sparse attention, grouped convolution.

Phase 10+ 에서:
- A (channel-as-node) foundations + 계층적 hybrid (그룹 내 채널 graph nest)
- DARTS / L0 / Gumbel-softmax 로 adjacency sparsification 학습
- Transformer 의 Q/K/V/O 통합
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

AdjInit = Literal["full", "identity"]


def _validate_groupable(features: int, group_size: int, name: str) -> int:
    if features % group_size != 0:
        raise ValueError(f"{name} ({features}) must be divisible by group_size ({group_size})")
    return features // group_size


class GroupGraphLinear(nn.Module):
    """Group-as-node graph linear layer.

    표준 ``nn.Linear`` 와 입출력 shape 동일하나, 내부적으로 ``group_size`` 단위 block 으로
    파라미터를 보관하고 그룹 간 routing 은 ``adj`` 가 결정.

    Args:
        in_features: 입력 차원 — ``group_size`` 의 배수여야 함.
        out_features: 출력 차원 — ``group_size`` 의 배수여야 함.
        group_size: 한 그룹의 채널 수. 표준 head_dim (64).
        adj_init: ``"full"`` 이면 adjacency 모두 1 (모든 그룹 routing 활성),
            ``"identity"`` 이면 동일 그룹 index 끼리만 1, 나머지 0 (block-diagonal — pure
            grouped operation, 가장 sparse 시작).

    Forward (입력 shape ``(..., in_features)`` → 출력 ``(..., out_features)``):
        x 를 ``(..., n_groups_in, group_size)`` 로 reshape →
        각 출력 group ``go`` 에 대해 ``y[go] = Σ_gi adj[go, gi] · (x[gi] @ W[go, gi])`` →
        ``(..., out_features)`` 로 reshape.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        *,
        adj_init: AdjInit = "full",
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups_in = _validate_groupable(in_features, group_size, "in_features")
        self.n_groups_out = _validate_groupable(out_features, group_size, "out_features")

        # block weights: shape (n_groups_out, n_groups_in, group_size, group_size)
        self.weight = nn.Parameter(
            torch.empty(self.n_groups_out, self.n_groups_in, group_size, group_size)
        )
        # adjacency: shape (n_groups_out, n_groups_in) — continuous routing scalar
        if adj_init == "full":
            adj = torch.ones(self.n_groups_out, self.n_groups_in)
        elif adj_init == "identity":
            adj = torch.eye(max(self.n_groups_out, self.n_groups_in))[
                : self.n_groups_out, : self.n_groups_in
            ].contiguous()
        else:
            raise ValueError(f"unknown adj_init: {adj_init}")
        self.adj = nn.Parameter(adj)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Kaiming uniform init on block weights — equivalent to standard Linear when all
        # blocks together form the full weight matrix. fan_in 은 in_features.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) → (..., n_groups_in, group_size)
        *batch, in_f = x.shape
        if in_f != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {in_f}")
        x_g = x.view(*batch, self.n_groups_in, self.group_size)

        # block matmul: for each (go, gi) compute x[gi] @ W[go, gi] → (..., go, gi, group_size)
        # einsum: ...gi,Goik (G=n_groups_out, g=n_groups_in 동일 index, i=in_chans, k=out_chans)
        # 결과 (..., G, g, k) — go 별 gi-routed contributions
        contrib = torch.einsum("...gi,Ggik->...Ggk", x_g, self.weight)
        # adjacency-weighted sum over gi: y[go] = Σ_gi adj[go, gi] · contrib[go, gi]
        y_g = torch.einsum("Gg,...Ggk->...Gk", self.adj, contrib)
        # reshape (..., n_groups_out, group_size) → (..., out_features)
        y = y_g.reshape(*batch, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def freeze_adjacency(self) -> None:
        """adjacency 를 학습 비대상으로 — Phase 10+ sparsification 학습 분리용."""
        self.adj.requires_grad_(False)

    def sparsify_adjacency(self, threshold: float) -> int:
        """절대값이 ``threshold`` 미만인 adjacency entry 를 0 으로 강제 (in-place).

        Phase 10+ 의 hard sparsification 학습 결과 적용을 위한 stub. Returns 비활성화된 entry 수.
        """
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        with torch.no_grad():
            mask = self.adj.abs() >= threshold
            n_zeroed = int((~mask).sum().item())
            self.adj.mul_(mask.to(self.adj.dtype))
        return n_zeroed

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, n_groups_in={self.n_groups_in}, "
            f"n_groups_out={self.n_groups_out}"
        )
