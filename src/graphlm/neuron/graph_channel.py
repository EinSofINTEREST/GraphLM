"""Phase 10 — channel-as-node graph hidden layer foundations.

Hidden_dim H 채널을 graph 의 node 로 다룬다 (paradigm 의 finest unit). standard ``nn.Linear``
의 weight matrix 의 각 entry 를 *edge* 로 재해석하고, 학습 가능 ``adj`` (per-edge gate) 를
도입.

```
in_features H_in 채널 = 입력 nodes
out_features H_out 채널 = 출력 nodes
edge[out, in]: 학습 가능 scalar adj[out, in]
effective weight: W_eff = adj ⊙ W (elementwise product)
forward: y = W_eff @ x = (adj * W) @ x
```

설계 의미:
- node = 1 채널 (graph 의 vertex)
- edge = (in, out) 쌍의 connection — adj 가 routing strength, W 가 transformation
- adj=full (모두 1) → effective = W → standard Linear forward 동치 (function preserving)
- adj=uniform_small → Phase 2 sweet spot 패턴 channel-level edge 에 적용 (0-init 회피)

**0-init 금지 규칙 적용** (rationale: Phase 9 결과 PR #60 +
[feedback_no_zero_init.md](https://www.notion.so/36ce8b70b7aa8100b0acf756686d2e9f) Notion 정리):
- Phase 1 dead block / Phase 7 amplitude vanishing / Phase 9 block-diagonal 의 3차 반복 발견
- 본 모듈은 ``adj_init="zero"`` 옵션 명시적 거부 (ValueError)
- default 권장 = ``"full"`` (function preserving) 또는 ``"uniform_small"`` (sweet spot 패턴)

Phase 9 (GroupGraphLinear) 와의 차이:
- Phase 9: group_size 단위 block 으로 묶어 (n_groups_out, n_groups_in) adjacency
- Phase 10: 채널 1개 단위 — (out, in) 전체 adjacency, paradigm 의 finest unit
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

AdjInit = Literal["full", "uniform_small", "uniform_around_one"]


class ChannelGraphLinear(nn.Module):
    """Channel-as-node graph linear layer (Phase 10 paradigm 의 finest unit foundation).

    Args:
        in_features: 입력 채널 수 (graph 의 in-nodes).
        out_features: 출력 채널 수 (graph 의 out-nodes).
        adj_init: 선택지 (memory: feedback_no_zero_init.md 의 0-init 금지 + magnitude rule):
            - ``"full"`` — 모두 1 (function preserving, standard Linear 와 forward 동치)
            - ``"uniform_around_one"`` — uniform[0.95, 1.05] (1.0 근처 small noise,
              weight multiplier 의 적정 magnitude + 0-init 회피 + adj 학습 활성, Phase 11+ 권장)
            - ``"uniform_small"`` — uniform[0.05, 0.15] (Phase 2 residual-gate sweet spot 패턴,
              **anti-pattern** — weight multiplier 위치엔 magnitude 가 작아 +0.18 열위, Phase 10 실측)
            - ❌ ``"zero"`` — 거부 (Phase 1/7/9 vanishing 함정 3차 재현, rationale: PR #60)
        bias: bias 사용 여부 (standard Linear 동일).

    Forward:
        ``y = (adj * weight) @ x + bias``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        adj_init: AdjInit = "full",
        bias: bool = True,
    ):
        super().__init__()
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(f"in_features must be a positive int, got {in_features!r}")
        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError(f"out_features must be a positive int, got {out_features!r}")

        self.in_features = in_features
        self.out_features = out_features

        # standard Linear init for W (fan_in = in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)

        # adj init — 0-init 거부 + magnitude rule (memory: feedback_no_zero_init.md)
        if adj_init == "full":
            adj = torch.ones(out_features, in_features)
        elif adj_init == "uniform_small":
            # Phase 2 residual-gate sweet spot (0.10) ± δ — *주의*: weight multiplier 위치에는
            # magnitude rule 위반 (effective_w 가 10% scale 로 줄어 +0.18 열위, Phase 10 발견).
            # 유지 이유: ablation 비교용 / 명시적 anti-pattern 데모.
            adj = torch.empty(out_features, in_features).uniform_(0.05, 0.15)
        elif adj_init == "uniform_around_one":
            # Phase 11 scale-corrected: weight multiplier 의 적정 magnitude ≈ 1.0 + small noise
            # (memory: feedback_no_zero_init.md 의 magnitude rule). 0-init 회피 + scale 균형.
            adj = torch.empty(out_features, in_features).uniform_(0.95, 1.05)
        elif adj_init in {"zero", "zeros"}:
            raise ValueError(
                f"adj_init={adj_init!r} 는 금지됨 — 0-init 은 vanishing gradient 함정 "
                "(Phase 1 dead block / Phase 7 amplitude vanishing / Phase 9 block-diagonal "
                "에서 3차 재현). rationale: Phase 9 PR #60. "
                "'full' (function preserving) 또는 'uniform_small' (sweet spot 패턴) 사용 권장."
            )
        else:
            raise ValueError(f"unknown adj_init: {adj_init!r}")
        self.adj = nn.Parameter(adj)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        effective_w = self.adj * self.weight
        return F.linear(x, effective_w, self.bias)

    def adj_sparsity(self, threshold: float = 0.05) -> float:
        """현재 |adj| < threshold 인 edge 의 비율 (0~1)."""
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        with torch.no_grad():
            below = (self.adj.abs() < threshold).float().mean().item()
        return float(below)

    def sparsify_adj(self, threshold: float) -> int:
        """|adj| < threshold 인 entry 를 0 으로 강제 (in-place). returns 비활성화된 entry 수.

        ``self.adj.data.masked_fill_`` 사용 — leaf+requires_grad tensor 에서 in-place op 의
        autograd 안전성 명시적 보장 (gemini #3301728271 방어적 패턴).
        """
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        with torch.no_grad():
            zero_mask = self.adj.abs() < threshold
            n_zeroed = int(zero_mask.sum().item())
            self.adj.data.masked_fill_(zero_mask, 0.0)
        return n_zeroed

    def freeze_adjacency(self) -> None:
        """adj 를 학습 비대상으로."""
        self.adj.requires_grad_(False)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"adj_shape={tuple(self.adj.shape)}"
        )
