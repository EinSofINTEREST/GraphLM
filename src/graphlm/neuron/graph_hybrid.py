"""Phase 12 — hierarchical hybrid graph hidden layer (outer group + inner channel).

사용자 vision (히든 레이어 = graph) 의 **ultimate 구조**: Phase 9 group-as-node 와
Phase 10/11 channel-as-node 의 계층적 결합.

```
hidden_dim H = G groups × k channels per group
weight W: (G_out, G_in, k, k)  # block-organized (Phase 9 동일)
adj_outer: (G_out, G_in)        # group-level routing (Phase 9 의 adj)
adj_inner: (G_out, G_in, k, k)  # channel-level fine-grained gate (Phase 10/11 의 채널 gate)

forward (block matmul + dual gates):
  contrib[go, gi] = (adj_inner[go, gi] * W[go, gi]) @ x[gi]    # shape (k,)
  y[go] = Σ_gi adj_outer[go, gi] · contrib[go, gi]              # shape (k,)
  y_flat = reshape(y, (G_out · k,))
```

effective edge weight (channel-pair) =
    adj_outer[group(out), group(in)] · adj_inner[out, in] · W[out, in]

설계 의미:
- **outer adj** = group-level *coarse routing* (어느 group 가 어느 group 에 연결)
- **inner adj** = channel-level *fine-grained gate* (각 connection 의 strength tuning)
- 둘 다 학습 가능 + 둘 다 magnitude rule 적용 (weight multiplier 위치 ≈ 1.0)
- function preservation: 둘 다 ``"full"`` (모두 1) → standard Linear 와 forward 동치

**0-init 금지 + magnitude rule** (memory: feedback_no_zero_init.md):
- adj_outer / adj_inner 모두 ``"zero"`` 거부 (ValueError)
- weight multiplier 위치 → sweet spot magnitude ≈ 1.0
- 옵션:
  - ``"full"`` (모두 1) — function preserving
  - ``"identity"`` (block-diagonal, n_groups_out == n_groups_in 만, adj_outer 만) — Phase 9 호환
  - ``"uniform_around_one"`` (uniform[0.95, 1.05]) — scale-corrected + 학습 활성

Phase 9 / 10 / 11 과의 관계:
- adj_outer=full + adj_inner=full == standard Linear (function preserving)
- adj_outer=full + adj_inner=uniform_around_one ≈ Phase 11 의 channel-level scale-corrected
- adj_outer=identity + adj_inner=full ≈ Phase 9 의 group_identity (block-diagonal)
- 모두 한 module 에서 통합 표현 가능
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

AdjOuterInit = Literal["full", "identity", "uniform_around_one"]
AdjInnerInit = Literal["full", "uniform_around_one"]


def _validate_groupable(features: int, group_size: int, name: str) -> int:
    if not isinstance(group_size, int) or group_size < 1:
        raise ValueError(f"group_size must be a positive int, got {group_size!r}")
    if features % group_size != 0:
        raise ValueError(f"{name} ({features}) must be divisible by group_size ({group_size})")
    return features // group_size


def _make_outer_adj(n_groups_out: int, n_groups_in: int, init: AdjOuterInit) -> Tensor:
    if init == "full":
        return torch.ones(n_groups_out, n_groups_in)
    if init == "identity":
        if n_groups_out != n_groups_in:
            raise ValueError(
                f"adj_outer_init='identity' requires square (n_groups_out=={n_groups_in}), "
                f"got out={n_groups_out} in={n_groups_in}"
            )
        return torch.eye(n_groups_out, n_groups_in)
    if init == "uniform_around_one":
        return torch.empty(n_groups_out, n_groups_in).uniform_(0.95, 1.05)
    if init in {"zero", "zeros"}:
        raise ValueError(
            f"adj_outer_init={init!r} 는 금지됨 — 0-init vanishing 함정. "
            "rationale: Phase 9 PR #60, magnitude rule: Phase 10 PR #62. "
            "'full' / 'identity' / 'uniform_around_one' 사용 권장."
        )
    raise ValueError(f"unknown adj_outer_init: {init!r}")


def _make_inner_adj(
    n_groups_out: int,
    n_groups_in: int,
    group_size: int,
    init: AdjInnerInit,
) -> Tensor:
    shape = (n_groups_out, n_groups_in, group_size, group_size)
    if init == "full":
        return torch.ones(shape)
    if init == "uniform_around_one":
        return torch.empty(shape).uniform_(0.95, 1.05)
    if init in {"zero", "zeros"}:
        raise ValueError(
            f"adj_inner_init={init!r} 는 금지됨 — 0-init vanishing 함정. "
            "rationale: Phase 9 PR #60, magnitude rule: Phase 10 PR #62. "
            "'full' 또는 'uniform_around_one' 사용 권장."
        )
    raise ValueError(f"unknown adj_inner_init: {init!r}")


class HybridGraphLinear(nn.Module):
    """Hierarchical hybrid graph linear: outer group + inner channel routing.

    paradigm 의 ultimate 단계 — Phase 9 group + Phase 10/11 channel 의 통합.

    Args:
        in_features, out_features: 표준 Linear shape (둘 다 group_size 의 배수)
        group_size: k (한 group 당 채널 수)
        adj_outer_init: ``"full"`` / ``"identity"`` / ``"uniform_around_one"``
        adj_inner_init: ``"full"`` / ``"uniform_around_one"``
        bias: bias 사용 여부

    Forward:
        ``y[go] = Σ_gi adj_outer[go, gi] · (adj_inner[go, gi] * W[go, gi]) @ x[gi]``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        *,
        adj_outer_init: AdjOuterInit = "full",
        adj_inner_init: AdjInnerInit = "full",
        bias: bool = True,
    ):
        super().__init__()
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(f"in_features must be a positive int, got {in_features!r}")
        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError(f"out_features must be a positive int, got {out_features!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups_in = _validate_groupable(in_features, group_size, "in_features")
        self.n_groups_out = _validate_groupable(out_features, group_size, "out_features")

        # block weight: shape (G_out, G_in, k, k)
        self.weight = nn.Parameter(
            torch.empty(self.n_groups_out, self.n_groups_in, group_size, group_size)
        )
        # standard Linear-equivalent init (fan_in = in_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)

        # outer adj (group-level routing)
        self.adj_outer = nn.Parameter(
            _make_outer_adj(self.n_groups_out, self.n_groups_in, adj_outer_init)
        )
        # inner adj (channel-level gate within each block)
        self.adj_inner = nn.Parameter(
            _make_inner_adj(self.n_groups_out, self.n_groups_in, group_size, adj_inner_init)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Phase 15: edge_mask — buffer (학습 X, state_dict 에 포함). 초기값 모두 1 (no prune).
        # prune_by_magnitude() 호출 시 일부 위치가 영구 0 으로 전환 → 해당 edge 의 forward
        # 기여도 0 + gradient chain 도 mask=0 위치에서 끊김 → resurrection 방지.
        self.register_buffer(
            "edge_mask",
            torch.ones(self.n_groups_out, self.n_groups_in, group_size, group_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) → (..., n_groups_in, group_size)
        *batch, in_f = x.shape
        if in_f != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {in_f}")
        x_g = x.reshape(*batch, self.n_groups_in, self.group_size)

        # 메모리 효율 최적화 (gemini #3302293739): adj_outer 와 adj_inner 를 weight 수준에서
        # 미리 결합 → (*batch, G_out, G_in, k) 의 큰 intermediate tensor 회피.
        # Phase 15: edge_mask 도 같은 위치에 곱함 → pruned edge 의 forward 0 + gradient 차단.
        eff_w = (
            self.adj_outer.unsqueeze(-1).unsqueeze(-1)
            * self.adj_inner
            * self.weight
            * self.edge_mask
        )
        # single einsum — output (*batch, G_out, k), 중간 (*batch, G_out, G_in, k) 텐서 없음
        y_g = torch.einsum("...gi,Ggik->...Gk", x_g, eff_w)
        # flatten back to (..., out_features)
        y = y_g.reshape(*batch, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def adj_outer_sparsity(self, threshold: float = 0.05) -> float:
        """|adj_outer| < threshold 인 group-edge 비율."""
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        with torch.no_grad():
            return float((self.adj_outer.abs() < threshold).float().mean().item())

    def adj_inner_sparsity(self, threshold: float = 0.05) -> float:
        """|adj_inner| < threshold 인 channel-edge 비율."""
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        with torch.no_grad():
            return float((self.adj_inner.abs() < threshold).float().mean().item())

    def freeze_adj_outer(self) -> None:
        self.adj_outer.requires_grad_(False)

    def freeze_adj_inner(self) -> None:
        self.adj_inner.requires_grad_(False)

    # ── Phase 15: edge prune ────────────────────────────────────

    def effective_edge_magnitude(self) -> Tensor:
        """현재 forward 에 적용되는 edge magnitude (shape: (G_out, G_in, k, k)).

        = ``|adj_outer · adj_inner · W| · edge_mask`` — 이미 pruned 된 edge 는 0.
        prune 결정 / sparsity 측정에 사용.
        """
        with torch.no_grad():
            return (
                self.adj_outer.unsqueeze(-1).unsqueeze(-1) * self.adj_inner * self.weight
            ).abs() * self.edge_mask

    def prune_by_magnitude(self, threshold: float) -> int:
        """edge magnitude < threshold 인 위치를 영구 prune (mask=0).

        gradient resurrection 방지: forward 의 ``eff_w *= edge_mask`` 곱셈으로
        해당 위치의 gradient 가 ``adj_outer``, ``adj_inner``, ``weight`` 모두에서 0 으로 차단됨.

        Args:
            threshold: |adj_outer · adj_inner · W| 의 절대값 임계값.

        Returns:
            이번 호출에서 신규로 prune 된 edge 수.
        """
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        with torch.no_grad():
            mag = self.effective_edge_magnitude()
            new_dead = (mag < threshold) & (self.edge_mask > 0)
            n_pruned = int(new_dead.sum().item())
            self.edge_mask[new_dead] = 0.0
        return n_pruned

    def prune_bottom_fraction(self, fraction: float) -> int:
        """살아있는 edge 중 magnitude 하위 ``fraction`` 비율을 prune.

        threshold 기반보다 target sparsity 제어가 용이. fraction=0.3 → 살아있는 edge 의 30% 추가 prune.

        Args:
            fraction: 0.0 ~ 1.0, 살아있는 edge 중 prune 할 비율.

        Returns:
            이번 호출에서 신규로 prune 된 edge 수.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        with torch.no_grad():
            mag = self.effective_edge_magnitude()
            alive_mask = self.edge_mask > 0
            alive_count = int(alive_mask.sum().item())
            n_to_prune = int(alive_count * fraction)
            if n_to_prune == 0:
                return 0
            # 살아있는 edge 만의 magnitude 중 하위 n_to_prune 위치
            alive_mag = mag[alive_mask]
            # kthvalue 는 sorted 의 k 번째 작은 값 (1-indexed)
            kth = alive_mag.kthvalue(n_to_prune).values.item()
            new_dead = (mag <= kth) & alive_mask
            # tie-breaking: kth 와 같은 magnitude 가 여럿이면 약간 초과될 수 있으나 OK
            n_pruned = int(new_dead.sum().item())
            self.edge_mask[new_dead] = 0.0
        return n_pruned

    def effective_sparsity(self) -> float:
        """전체 edge 중 영구 prune (mask=0) 비율."""
        with torch.no_grad():
            return float((self.edge_mask == 0).float().mean().item())

    def n_alive_edges(self) -> int:
        """살아있는 edge 수 (mask=1)."""
        with torch.no_grad():
            return int((self.edge_mask > 0).sum().item())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, n_groups_in={self.n_groups_in}, "
            f"n_groups_out={self.n_groups_out}"
        )
