"""Neuron paradigm Phase 1 — Attention module 단위 dynamic add backbone.

핵심 설계 차이 (기존 `graphlm.models.backbone.GrowingDecoder` 대비):
- 기존: block 단위 동적 추가 (depth growth). block = LN + attn + LN + FFN + scalar α
- 본 module: **block 안에 multiple parallel attention modules** 가 학습 중 동적 추가.
  각 attention module 이 1개 노드 (function 단위). LayerNorm + CausalSelfAttention
  + 학습 가능 α 로 구성.

Phase 1 scope:
- block count 고정 (n_layers), block 안의 attention module 수만 동적 (n_attn growing)
- FFN 은 표준 1개 유지 (Phase 2 이후에서 FFN 도 multiple 로 확장 검토)
- function preservation: 새 attention module 을 α=0 으로 추가하면 forward 동일

기존 GraphLM Phase 1 의 dead block 패턴이 head-level (정확히는 attention module level)
에서도 재현되는지 검증하는 것이 첫 가설.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NeuronConfig:
    """NeuronGrowingDecoder 의 정적 hyperparameter."""

    vocab_size: int
    hidden_dim: int = 256
    n_heads: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0
    # block 수 (고정)
    n_layers: int = 4
    # 각 block 의 초기 attention module 수
    n_init_attn: int = 1
    # Phase 4: α 를 scalar (False, Phase 1~3 호환) 또는 per-channel vector ∈ ℝ^{hidden_dim} (True)
    alpha_per_channel: bool = False
    # Phase 6: α 를 위치의 함수 α(t)[c] = a_c·sin(t·ω_c) + b_c 로 정의 (SinusoidalAlpha).
    # alpha_per_channel 과 동시 True 불가 — exclusive.
    alpha_positional: bool = False

    def __post_init__(self) -> None:
        # downstream divide-by-zero / 잘못된 attention head dim 방지 위한 최소 검증
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.n_init_attn < 1:
            raise ValueError(f"n_init_attn must be >= 1, got {self.n_init_attn}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} not divisible by n_heads {self.n_heads}"
            )
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.alpha_per_channel and self.alpha_positional:
            raise ValueError(
                "alpha_per_channel and alpha_positional are mutually exclusive (got both True)"
            )


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention — `graphlm.models.backbone.CausalSelfAttention` 동일."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FFN(nn.Module):
    """Standard 2-layer FFN with GELU."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(torch.nn.functional.gelu(self.fc1(x))))


class SinusoidalAlpha(nn.Module):
    """Phase 6: 위치의 함수로 정의되는 per-channel α.

    α(t)[c] = a_c · sin(t · ω_c) + b_c

    - ``amplitude`` (a_c): 채널별 진폭 — position 의존성 크기. 0 이면 position 무관 (b_c constant).
    - ``log_freq`` (= log ω_c): 채널별 주파수 (양수 보장 위해 log-space). init 은 log-spaced
      [1/100, 1.0].
    - ``bias`` (b_c): 채널별 baseline. Phase 4/5 의 per-channel scalar α 와 등가 init.

    Init 규약 (function preservation 지원):
    - ``init_amp=0, init_bias=0`` → α(t) = 0 ∀t (forward 불변)
    - ``init_amp=0, init_bias=v`` → α(t) = v ∀t (Phase 4/5 sweet spot 등가)
    - 학습이 amplitude 를 키워 position dependency 가 자율적으로 emerge.

    forward 는 ``T`` (시퀀스 길이) 받고 ``(T, hidden_dim)`` 텐서 반환 — (B, T, H) attn 출력과
    broadcasting.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        init_amp: float = 0.0,
        init_bias: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.amplitude = nn.Parameter(
            torch.full((hidden_dim,), init_amp, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.full((hidden_dim,), init_bias, device=device, dtype=dtype))
        # log-spaced initial frequencies in [1/100, 1.0] → period [2π, 200π]
        log_freq_init = torch.linspace(
            math.log(0.01), math.log(1.0), hidden_dim, device=device, dtype=dtype
        )
        self.log_freq = nn.Parameter(log_freq_init)

    def forward(self, seq_len: int) -> Tensor:
        # device/dtype 는 학습된 파라미터로부터 (model.to() 이동에 따라 자동 일치)
        device = self.amplitude.device
        dtype = self.amplitude.dtype
        t = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(-1)  # (T, 1)
        freq = self.log_freq.exp().unsqueeze(0)  # (1, hidden_dim)
        return self.amplitude.unsqueeze(0) * torch.sin(t * freq) + self.bias.unsqueeze(0)


class NeuronBlock(nn.Module):
    """Pre-LN Decoder block with **multiple parallel attention modules** (n_attn growing).

    Forward:
        for each attention module i: x = x + alpha_i * attn_i(ln_attn_i(x))
        x = x + alpha_ffn * ffn(ln_ffn(x))

    각 attention module (= 1 노드) 는 독립적 LayerNorm + CausalSelfAttention + α.
    α 의 형태는 ``NeuronConfig`` 의 두 플래그로 결정 (mutually exclusive):
    - 기본 (둘 다 False, Phase 1~3): scalar (broadcasting 으로 적용)
    - ``alpha_per_channel=True`` (Phase 4+): per-channel vector ∈ ℝ^{hidden_dim}
    - ``alpha_positional=True`` (Phase 6+): SinusoidalAlpha 모듈, α(t)[c] = a_c·sin(t·ω_c) + b_c
    forward 는 isinstance(alpha, nn.Module) 로 dispatch.
    FFN 은 표준 1개 (Phase 1 scope 단순화).
    """

    def __init__(self, cfg: NeuronConfig, n_init_attn: int = 1):
        super().__init__()
        self.cfg = cfg
        self.attn_lns = nn.ModuleList([nn.LayerNorm(cfg.hidden_dim) for _ in range(n_init_attn)])
        self.attns = nn.ModuleList(
            [
                CausalSelfAttention(cfg.hidden_dim, cfg.n_heads, cfg.dropout)
                for _ in range(n_init_attn)
            ]
        )
        # alpha container 타입은 alpha_positional 에 따라 분기:
        # - positional: ModuleList[SinusoidalAlpha] (forward 시 alpha(T) 호출)
        # - 그 외: ParameterList[nn.Parameter] (직접 broadcast)
        if cfg.alpha_positional:
            self.attn_alphas = nn.ModuleList(
                [self._make_alpha_module(1.0) for _ in range(n_init_attn)]
            )
            self.ffn_alpha = self._make_alpha_module(1.0)
        else:
            self.attn_alphas = nn.ParameterList(
                [nn.Parameter(self._make_alpha_tensor(1.0)) for _ in range(n_init_attn)]
            )
            self.ffn_alpha = nn.Parameter(self._make_alpha_tensor(1.0))
        self.ln_ffn = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_dim, cfg.dropout)

    def _make_alpha_tensor(
        self,
        value: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """scalar (Phase 1~3) or per-channel vector ∈ ℝ^{hidden_dim} (Phase 4+).

        device/dtype 를 명시하면 그대로 사용 (add_attn 시 기존 param 과 동기). None 이면 default
        (__init__ 호출 — 이후 ``model.to()`` 가 일괄 이동).
        """
        if self.cfg.alpha_per_channel:
            return torch.full((self.cfg.hidden_dim,), value, device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    def _make_alpha_module(
        self,
        value: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SinusoidalAlpha":
        """Phase 6: ``init_amp=0, init_bias=value`` — sweet spot 등가 init.

        amplitude 가 0 으로 시작하므로 init 시점 forward 는 per-channel α=value 와 동일.
        학습이 amplitude 를 키워 position dependency 가 emerge.
        """
        return SinusoidalAlpha(
            self.cfg.hidden_dim,
            init_amp=0.0,
            init_bias=value,
            device=device,
            dtype=dtype,
        )

    @property
    def n_attn(self) -> int:
        return len(self.attns)

    def forward(self, x: Tensor) -> Tensor:
        # parallel: 모든 attention module 이 *같은* 입력 (attn_input) 에서 계산.
        # block 내부 depth 가 늘어나지 않고 width-like (head-level parallel) 의미 유지.
        T = x.shape[1]
        attn_input = x
        for ln, attn, alpha in zip(self.attn_lns, self.attns, self.attn_alphas, strict=True):
            a = alpha(T) if isinstance(alpha, nn.Module) else alpha
            x = x + a * attn(ln(attn_input))
        ffn_a = self.ffn_alpha(T) if isinstance(self.ffn_alpha, nn.Module) else self.ffn_alpha
        return x + ffn_a * self.ffn(self.ln_ffn(x))

    def add_attn(self, *, residual_scale: float = 0.0, init_std: float | None = None) -> int:
        """Append a new attention module (LayerNorm + CausalSelfAttention + α).

        새 α 는 ``cfg.alpha_per_channel`` 에 따라 scalar (Phase 1~3 호환) 또는 per-channel
        vector ∈ ℝ^{hidden_dim} (Phase 4+) 로 생성. 어느 경우든 ``residual_scale`` (float) 는
        모든 채널에 uniform init.

        Args:
            residual_scale: 신규 attention 의 초기 α. function preservation 보장을 위해 default 0.0.
            init_std: weight init 의 std. None 이면 caller (decoder) 가 정한 std.

        Returns:
            새 attention 의 index (0-based).
        """
        new_ln = nn.LayerNorm(self.cfg.hidden_dim)
        new_attn = CausalSelfAttention(self.cfg.hidden_dim, self.cfg.n_heads, self.cfg.dropout)
        if init_std is not None:
            for m in new_attn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=init_std)
        # 기존 param 의 device/dtype 와 동기 — Parameter().to(device) 가 Tensor 를 반환할 수
        # 있는 PyTorch quirk 회피 (gemini #3296196217).
        ref_param = next(self.parameters())
        if self.cfg.alpha_positional:
            new_alpha = self._make_alpha_module(
                residual_scale, device=ref_param.device, dtype=ref_param.dtype
            )
        else:
            new_alpha = nn.Parameter(
                self._make_alpha_tensor(
                    residual_scale, device=ref_param.device, dtype=ref_param.dtype
                )
            )
        self.attn_lns.append(new_ln.to(ref_param.device))
        self.attns.append(new_attn.to(ref_param.device))
        self.attn_alphas.append(new_alpha)
        return len(self.attns) - 1


class NeuronGrowingDecoder(nn.Module):
    """Decoder Transformer with per-block multiple parallel attention modules growing.

    구조:
        token_emb + pos_emb → [NeuronBlock × n_layers] → final_ln → tied LM head

    block 수 (n_layers) 는 고정. 각 block 안의 attention module 수만 학습 중 동적.
    """

    def __init__(self, cfg: NeuronConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [NeuronBlock(cfg, n_init_attn=cfg.n_init_attn) for _ in range(cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        self._init_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self._init_std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def add_attn(self, block_idx: int, *, residual_scale: float = 0.0) -> int:
        """학습 중 특정 block 에 새 attention module 추가.

        Args:
            block_idx: 어느 block 에 추가할지 (0-based).
            residual_scale: 신규 attention 의 초기 α (default 0.0, function preservation).

        Returns:
            추가된 attention 의 index within the block.
        """
        if not 0 <= block_idx < len(self.blocks):
            raise IndexError(f"block_idx {block_idx} out of range [0, {len(self.blocks)})")
        return self.blocks[block_idx].add_attn(
            residual_scale=residual_scale, init_std=self._init_std
        )

    @property
    def n_attn_per_block(self) -> list[int]:
        return [b.n_attn for b in self.blocks]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward returning logits over vocab.

        Args:
            input_ids: shape ``[B, T]`` of token ids.

        Returns:
            logits of shape ``[B, T, vocab_size]``.
        """
        B, T = input_ids.shape
        if self.cfg.max_seq_len < T:
            raise ValueError(f"seq_len {T} exceeds max {self.cfg.max_seq_len}")
        pos = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return x @ self.token_emb.weight.t()
