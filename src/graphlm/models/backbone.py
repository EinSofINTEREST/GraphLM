"""Small Decoder Transformer with **dynamic depth** support (Phase 1).

핵심 design choices (38편 자료 기반):
- Decoder-only autoregressive LM (Llama/Mistral 스타일)
- Embedding sharing (input + LM head, MobileLLM)
- Pre-LayerNorm + GELU (modern default)
- **Per-block residual scale (alpha)** — Net2DeeperNet identity init 지원.
  새 block 을 alpha=0 으로 삽입하면 forward 가 변하지 않음 (function-preserving).
- ModuleList 로 layer 구성 → `add_block()` 으로 학습 중 동적 append 가능.

Phase 1 scope:
- depth 만 동적 (hidden / head / FFN dim 고정)
- GQA / nested FFN 등 후속 phase 의 elastic 은 미포함
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BackboneConfig:
    """Decoder Transformer 의 정적 hyperparameter."""

    vocab_size: int
    hidden_dim: int = 256
    n_heads: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0
    n_init_layers: int = 4


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention."""

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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # PyTorch ≥ 2.0 의 sdpa: causal mask + dropout 처리
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


class DecoderBlock(nn.Module):
    """Pre-LN Decoder block with **learnable residual scale** (alpha).

    alpha 가 attribute 가 아닌 Parameter 가 아닌 buffer 인 이유:
    Net2DeeperNet 의 \"identity init\" 은 alpha = 0 으로 시작해서 학습 가능하게 변화.
    Parameter 이지만 학습 trajectory 추적이 가능해야 하므로 buffer 가 아닌 Parameter.

    구체적으로:
    - 신규 block 삽입 직후: alpha = 0 → forward 동일 (function-preserving)
    - 학습 진행: alpha 가 의미 있는 값으로 학습됨
    """

    def __init__(self, cfg: BackboneConfig, residual_scale: float = 1.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg.hidden_dim, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_dim, cfg.dropout)
        # 학습 가능 residual scale — 새 block 은 0 으로 시작 (function-preserving)
        self.alpha = nn.Parameter(torch.tensor(residual_scale))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.alpha * self.attn(self.ln1(x))
        return x + self.alpha * self.ffn(self.ln2(x))


class GrowingDecoder(nn.Module):
    """Decoder Transformer 의 **layer 수가 학습 중 변동 가능** 한 backbone.

    구조:
        token_emb → pos_emb → [DecoderBlock × L] → final_ln → lm_head (weight-tied)
    여기서 L 은 학습 진행에 따라 증가 가능.

    Embedding sharing: lm_head 의 weight = token_emb 의 weight (MobileLLM 선택).
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg, residual_scale=1.0) for _ in range(cfg.n_init_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        # lm_head 는 token_emb weight 와 공유 — forward 시 token_emb.weight 를 그대로 사용
        # (별도 nn.Linear 를 두지 않고 functional 로 처리)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(
                    m.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_init_layers)
                )
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def add_block(self, *, residual_scale: float = 0.0) -> int:
        """학습 중 새 DecoderBlock 을 끝에 append.

        Args:
            residual_scale: 신규 block 의 초기 alpha. Net2DeeperNet 의 function preservation 을
                위해 default 0.0 (forward 변화 없음).

        Returns:
            새 block 의 index (0-based).

        Note:
            optimizer state 갱신은 caller 책임 — `loop.py` 의 grow callback 에서 처리.
        """
        new_block = DecoderBlock(self.cfg, residual_scale=residual_scale)
        # 같은 device 로 이동
        device = next(self.parameters()).device
        new_block = new_block.to(device)
        self.blocks.append(new_block)
        return len(self.blocks) - 1

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

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
        # weight tied LM head
        return x @ self.token_emb.weight.t()
