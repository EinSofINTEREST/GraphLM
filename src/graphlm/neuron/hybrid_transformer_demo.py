"""Phase 13 — HybridGraphTransformerLM demo + train helper.

노트북 분리 규약 준수. Phase 13 노트북 12-phase13-hybrid-transformer.ipynb 에서 import.

4 arch 비교 (모두 0-init 금지 + magnitude rule 적용):
- ``"plain"`` — RMSNorm + 표준 nn.Linear FFN (baseline)
- ``"hybrid_full_full"`` — outer=full + inner=full (function preserving)
- ``"hybrid_full_around_one"`` — outer=full + inner=uniform_around_one (Phase 11 channel-level)
- ``"hybrid_around_one_around_one"`` — 둘 다 uniform_around_one (fully scale-corrected)

identity outer 는 Phase 13 에서 미지원 — FFN 의 rectangular 구조상 의미 없음.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.data.tinyshakespeare import TinyShakespeareDataset, iter_random_batches
from graphlm.neuron.hybrid_transformer import (
    Arch,
    HybridGraphTransformerBlock,
    make_block,
)
from graphlm.neuron.rms_norm import RMSNorm
from graphlm.utils import set_seed


class HybridGraphTransformerLM(nn.Module):
    """Small char-LM Transformer with arch-dispatched FFN.

    - token embedding + learned positional embedding
    - N × pre-norm block (RMSNorm + attn + RMSNorm + FFN)
    - final RMSNorm + LM head (weight not tied for 공정 비교)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        max_seq_len: int,
        arch: Arch,
        group_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.arch = arch
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                make_block(
                    arch,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    group_size=group_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _batch, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        pos = torch.arange(seq_len, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.lm_head(h)


def _block_iter(model: HybridGraphTransformerLM) -> Iterator[HybridGraphTransformerBlock]:
    """모델의 hybrid block 만 yield (plain 은 skip)."""
    for blk in model.blocks:
        if isinstance(blk, HybridGraphTransformerBlock):
            yield blk


def _snapshot_adj(model: HybridGraphTransformerLM) -> list[dict[str, dict[str, Tensor]]] | None:
    """hybrid arch 인 경우 각 block 의 FFN adj snapshot (Phase 12 demo 와 동일 hierarchy)."""
    if model.arch == "plain":
        return None
    snapshots: list[dict[str, dict[str, Tensor]]] = []
    for blk in _block_iter(model):
        snapshots.append(
            {
                "fc1": {
                    "outer": blk.ffn.fc1.adj_outer.detach().cpu().clone(),
                    "inner": blk.ffn.fc1.adj_inner.detach().cpu().clone(),
                },
                "fc2": {
                    "outer": blk.ffn.fc2.adj_outer.detach().cpu().clone(),
                    "inner": blk.ffn.fc2.adj_inner.detach().cpu().clone(),
                },
            }
        )
    return snapshots


def train_hybrid_transformer_lm(
    *,
    dataset: TinyShakespeareDataset,
    vocab_size: int,
    seed: int,
    arch: Arch,
    hidden_dim: int,
    n_heads: int,
    ffn_dim: int,
    n_layers: int,
    group_size: int,
    block_size: int,
    batch_size: int,
    lr: float,
    max_steps: int,
    device: str = "cpu",
    dropout: float = 0.0,
) -> dict:
    """1 run 학습 — Phase 13 sweep unit.

    Returns: ``losses``, ``final_loss`` (last 100 mean), ``final_adj``
    (hybrid arch 인 경우 block 별 fc1/fc2 outer/inner snapshot list).
    """
    set_seed(seed)
    model = HybridGraphTransformerLM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        n_layers=n_layers,
        max_seq_len=block_size,
        arch=arch,
        group_size=group_size,
        dropout=dropout,
    ).to(device)
    data_iter = iter_random_batches(
        dataset, batch_size=batch_size, block_size=block_size, seed=seed
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _step in range(1, max_steps + 1):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    n_last = min(100, len(losses))
    final_loss = sum(losses[-n_last:]) / n_last if n_last > 0 else 0.0

    return {
        "losses": losses,
        "final_loss": final_loss,
        "final_adj": _snapshot_adj(model),
    }


def count_parameters(model: nn.Module) -> int:
    """전체 학습 가능 파라미터 수 (arch 간 공정 비교용 reporting)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "HybridGraphTransformerLM",
    "count_parameters",
    "train_hybrid_transformer_lm",
]
