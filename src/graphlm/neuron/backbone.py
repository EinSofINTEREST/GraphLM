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
    n_layers: int = 4
    """block 수 (고정)."""
    n_init_attn: int = 1
    """각 block 의 초기 attention module 수."""


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


class NeuronBlock(nn.Module):
    """Pre-LN Decoder block with **multiple parallel attention modules** (n_attn growing).

    Forward:
        for each attention module i: x = x + alpha_i * attn_i(ln_attn_i(x))
        x = x + alpha_ffn * ffn(ln_ffn(x))

    각 attention module (= 1 노드) 는 독립적 LayerNorm + CausalSelfAttention + scalar α.
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
        self.attn_alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(n_init_attn)]
        )
        self.ln_ffn = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_dim, cfg.dropout)
        self.ffn_alpha = nn.Parameter(torch.tensor(1.0))

    @property
    def n_attn(self) -> int:
        return len(self.attns)

    def forward(self, x: Tensor) -> Tensor:
        for ln, attn, alpha in zip(self.attn_lns, self.attns, self.attn_alphas, strict=True):
            x = x + alpha * attn(ln(x))
        return x + self.ffn_alpha * self.ffn(self.ln_ffn(x))

    def add_attn(self, *, residual_scale: float = 0.0, init_std: float | None = None) -> int:
        """Append a new attention module (LayerNorm + CausalSelfAttention + scalar α).

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
        new_alpha = nn.Parameter(torch.tensor(residual_scale))
        device = next(self.parameters()).device
        self.attn_lns.append(new_ln.to(device))
        self.attns.append(new_attn.to(device))
        # nn.ParameterList 에 추가
        self.attn_alphas.append(new_alpha.to(device))
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
