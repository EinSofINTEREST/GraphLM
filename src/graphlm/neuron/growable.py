"""Phase 8 — structural axis: nn.Parameter 정적 shape 초월 foundations.

PyTorch ``nn.Parameter`` 는 ``__init__`` 시점에 고정 shape 으로 생성되어 학습 중 shape 변경이
불가능하다. 이는 training-time dynamic parameter count paradigm 의 가장 큰 제약. 본 모듈은
해결책으로 다음을 제공:

1. **GrowableLinear / GrowableLayerNorm / GrowableEmbedding** — 표준 모듈과 호환되면서
   ``expand_*`` 메서드로 in/out features 를 학습 중 확장 가능.
2. **AdamW state 확장** — Parameter 교체와 동시에 optimizer 의 momentum (m) / variance (v) 도
   확장 (기존 dim 보존, 새 dim zero init). 학습 continuity 보장.
3. **Function preservation** — 신규 dim 을 zero 또는 identity-style 로 init 하여 expansion 직후
   forward output 불변 보장 (Net2Net / bert2BERT 철학).

참고: bert2BERT (Chen et al. ACL 2022), LiGO (Wang et al. ICLR 2023), MSG (Yuan et al.
NeurIPS 2023) 의 expansion 메커니즘과 동일 패러다임.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

InitMode = Literal["zero", "normal"]


def _replace_param_with_optimizer_state(
    module: nn.Module,
    attr: str,
    new_data: Tensor,
    optimizer: torch.optim.Optimizer | None,
    expand_axes: dict[int, int],
) -> nn.Parameter:
    """Module 의 Parameter 속성을 새 shape 의 Parameter 로 교체 + AdamW state 동기 확장.

    Args:
        module: 대상 모듈 (예: GrowableLinear).
        attr: 속성 이름 (예: ``"weight"``, ``"bias"``).
        new_data: 새 Parameter 의 data (이미 적절히 init 된 tensor).
        optimizer: 주어지면 ``optimizer.state[old_param]`` 의 m/v 도 확장하여 새 param 에
            연결. None 이면 state 무시 (학습 시작 전 또는 별도 관리).
        expand_axes: ``{axis: delta}`` — 어느 축이 얼마만큼 확장됐는지. AdamW state 의 m/v 도
            동일 shape 으로 확장 (새 entries zero).

    Returns:
        새로 생성한 ``nn.Parameter`` (module 의 attr 도 이미 교체됨).
    """
    old_param: nn.Parameter = getattr(module, attr)
    new_param = nn.Parameter(new_data, requires_grad=old_param.requires_grad)
    # Module 의 _parameters dict 갱신 — setattr 이 ParameterDict 처리 트리거
    setattr(module, attr, new_param)

    if optimizer is None:
        return new_param

    # optimizer.state[old_param] → 확장된 state 를 새 param 에 연결
    if old_param in optimizer.state:
        old_state = optimizer.state[old_param]
        new_state: dict = {}
        for key, value in old_state.items():
            if isinstance(value, Tensor) and value.shape == old_param.shape:
                # m / v 처럼 param-shape tensor → 확장
                expanded = _expand_tensor_with_zeros(value, expand_axes)
                new_state[key] = expanded
            else:
                # step counter 등 scalar / 기타 — 그대로 복사
                new_state[key] = value
        # 기존 entry 제거 + 새 param 으로 연결
        del optimizer.state[old_param]
        optimizer.state[new_param] = new_state

    # param_groups 에서 old_param → new_param 치환
    for group in optimizer.param_groups:
        params = group["params"]
        for i, p in enumerate(params):
            if p is old_param:
                params[i] = new_param

    return new_param


def _expand_tensor_with_zeros(t: Tensor, expand_axes: dict[int, int]) -> Tensor:
    """주어진 axis 들을 delta 만큼 확장 (새 entries = 0). 기존 entries 보존."""
    new_shape = list(t.shape)
    for axis, delta in expand_axes.items():
        new_shape[axis] += delta
    out = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
    # 기존 entries 복사
    slices = tuple(slice(0, s) for s in t.shape)
    out[slices] = t
    return out


def _init_new_block(shape: tuple[int, ...], mode: InitMode, std: float = 0.02) -> Tensor:
    if mode == "zero":
        return torch.zeros(shape)
    if mode == "normal":
        return torch.empty(shape).normal_(mean=0.0, std=std)
    raise ValueError(f"unknown init mode: {mode}")


class GrowableLinear(nn.Module):
    """nn.Linear 호환 + ``expand_out`` / ``expand_in`` 메서드 제공.

    forward 는 표준 linear (``x @ W^T + b``). expansion 후 AdamW state 도 함께 확장하면 학습
    continuity 보장.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def expand_out(
        self,
        delta: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        init: InitMode = "zero",
        std: float = 0.02,
    ) -> None:
        """Out features 를 ``delta`` 만큼 확장.

        weight: shape ``(out, in)`` → ``(out + delta, in)``. 새 rows 는 ``init`` 모드대로.
        bias: shape ``(out,)`` → ``(out + delta,)``. 새 entries zero (function preservation).

        ``init="zero"`` + zero bias → expansion 직후 새 out dim 의 출력 = 0 으로 function preserving.
        downstream layer 의 ``expand_in`` 이 같은 delta 로 따라오면 전체 forward 불변.
        """
        device = self.weight.device
        dtype = self.weight.dtype
        new_rows = _init_new_block((delta, self.in_features), init, std).to(
            device=device, dtype=dtype
        )
        new_weight = torch.cat([self.weight.data, new_rows], dim=0)
        _replace_param_with_optimizer_state(
            self, "weight", new_weight, optimizer, expand_axes={0: delta}
        )
        if self.bias is not None:
            new_bias_tail = torch.zeros(delta, device=device, dtype=dtype)
            new_bias = torch.cat([self.bias.data, new_bias_tail], dim=0)
            _replace_param_with_optimizer_state(
                self, "bias", new_bias, optimizer, expand_axes={0: delta}
            )
        self.out_features += delta

    def expand_in(
        self,
        delta: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        init: InitMode = "zero",
        std: float = 0.02,
    ) -> None:
        """In features 를 ``delta`` 만큼 확장.

        weight: shape ``(out, in)`` → ``(out, in + delta)``. 새 columns 는 ``init`` 모드대로.

        ``init="zero"`` → 새 입력 dim 이 출력에 기여 = 0. upstream layer 의 ``expand_out`` 이
        같은 delta 로 새 출력 dim 을 만들었다면 전체 forward 불변 (둘 모두 0 으로 이어짐).
        bias 는 in 축과 무관 — 변경 없음.
        """
        device = self.weight.device
        dtype = self.weight.dtype
        new_cols = _init_new_block((self.out_features, delta), init, std).to(
            device=device, dtype=dtype
        )
        new_weight = torch.cat([self.weight.data, new_cols], dim=1)
        _replace_param_with_optimizer_state(
            self, "weight", new_weight, optimizer, expand_axes={1: delta}
        )
        self.in_features += delta


class GrowableLayerNorm(nn.Module):
    """nn.LayerNorm 의 확장 가능 버전 — ``normalized_shape`` 이 마지막 dim 인 경우만 지원.

    expand 시: weight 새 entries = 1.0, bias 새 entries = 0.0 → identity-preserving.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(
            x, (self.normalized_shape,), self.weight, self.bias, self.eps
        )

    def expand(self, delta: int, *, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Normalized dim 을 ``delta`` 만큼 확장 (new weight=1, new bias=0)."""
        device = self.weight.device
        dtype = self.weight.dtype
        new_w_tail = torch.ones(delta, device=device, dtype=dtype)
        new_b_tail = torch.zeros(delta, device=device, dtype=dtype)
        new_weight = torch.cat([self.weight.data, new_w_tail], dim=0)
        new_bias = torch.cat([self.bias.data, new_b_tail], dim=0)
        _replace_param_with_optimizer_state(
            self, "weight", new_weight, optimizer, expand_axes={0: delta}
        )
        _replace_param_with_optimizer_state(
            self, "bias", new_bias, optimizer, expand_axes={0: delta}
        )
        self.normalized_shape += delta


class GrowableEmbedding(nn.Module):
    """nn.Embedding 의 확장 가능 버전 — ``embedding_dim`` 만 확장 (vocab 고정).

    expand 시: 새 dim 의 embedding 값 zero (function preservation — 새 dim 의 정보가 downstream
    에 0 으로 흘러감, downstream 의 expand_in 이 따라오면 전체 forward 불변).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.embedding(x, self.weight)

    def expand_dim(
        self,
        delta: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        init: InitMode = "zero",
        std: float = 0.02,
    ) -> None:
        device = self.weight.device
        dtype = self.weight.dtype
        new_cols = _init_new_block((self.num_embeddings, delta), init, std).to(
            device=device, dtype=dtype
        )
        new_weight = torch.cat([self.weight.data, new_cols], dim=1)
        _replace_param_with_optimizer_state(
            self, "weight", new_weight, optimizer, expand_axes={1: delta}
        )
        self.embedding_dim += delta
