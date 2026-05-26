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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.data.tinyshakespeare import TinyShakespeareDataset, iter_random_batches
from graphlm.neuron.graph_hybrid import HybridGraphLinear
from graphlm.neuron.hybrid_transformer import (
    Arch,
    FullGraphTransformerBlock,
    HybridGraphTransformerBlock,
    make_block,
    make_full_block,
)
from graphlm.neuron.rms_norm import RMSNorm
from graphlm.utils import set_seed


@dataclass(frozen=True)
class HybridTransformerTrainConfig:
    """1 run 학습에 필요한 모든 hyperparameter (CodeRabbit #3303186824 — 14 args 통합).

    구조: model / data / optim / runtime 4 그룹으로 묶어서 가독성 ↑.
    """

    # data
    dataset: TinyShakespeareDataset
    vocab_size: int

    # model
    hidden_dim: int
    n_heads: int
    ffn_dim: int
    n_layers: int
    group_size: int
    arch: Arch
    dropout: float = 0.0
    # Phase 14: True → make_full_block (attention + FFN 둘 다 graph),
    # False → make_block (FFN-only graph, Phase 13 default)
    use_full_graph: bool = False

    # train
    block_size: int = 64
    batch_size: int = 32
    lr: float = 3e-4
    max_steps: int = 1500

    # Phase 15: edge prune (one-shot at midpoint by default)
    # prune_at_step=None → no prune. 0 < step ≤ max_steps 면 해당 step 끝에서 prune 실행.
    prune_at_step: int | None = None
    # 살아있는 edge 중 하위 magnitude 비율 (prune_bottom_fraction 사용). 0.0 → no-op.
    prune_fraction: float = 0.0

    # runtime
    seed: int = 0
    device: str = "cpu"


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
        use_full_graph: bool = False,
    ):
        super().__init__()
        self.arch = arch
        self.use_full_graph = use_full_graph
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        block_factory = make_full_block if use_full_graph else make_block
        self.blocks = nn.ModuleList(
            [
                block_factory(
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


def _block_iter(
    model: HybridGraphTransformerLM,
) -> Iterator[HybridGraphTransformerBlock | FullGraphTransformerBlock]:
    """모델의 hybrid block 만 yield (plain 은 skip). Phase 13 + Phase 14 둘 다 지원."""
    for blk in model.blocks:
        if isinstance(blk, HybridGraphTransformerBlock | FullGraphTransformerBlock):
            yield blk


def _snapshot_layer(layer) -> dict[str, Tensor]:
    """HybridGraphLinear 한 layer 의 outer/inner snapshot."""
    return {
        "outer": layer.adj_outer.detach().cpu().clone(),
        "inner": layer.adj_inner.detach().cpu().clone(),
    }


def _snapshot_adj(model: HybridGraphTransformerLM) -> list[dict[str, dict[str, Tensor]]] | None:
    """hybrid arch 인 경우 각 block 의 adj snapshot.

    - Phase 13 (FFN-only graph): ``{"fc1", "fc2"}``
    - Phase 14 (full graph): ``{"qkv", "out", "fc1", "fc2"}`` — attention adj 까지 포함
    """
    if model.arch == "plain":
        return None
    snapshots: list[dict[str, dict[str, Tensor]]] = []
    for blk in _block_iter(model):
        snap: dict[str, dict[str, Tensor]] = {
            "fc1": _snapshot_layer(blk.ffn.fc1),
            "fc2": _snapshot_layer(blk.ffn.fc2),
        }
        if isinstance(blk, FullGraphTransformerBlock):
            snap["qkv"] = _snapshot_layer(blk.attn.qkv)
            snap["out"] = _snapshot_layer(blk.attn.out)
        snapshots.append(snap)
    return snapshots


def _prune_model(model: nn.Module, fraction: float) -> dict[str, int]:
    """모델의 모든 HybridGraphLinear 에 prune_bottom_fraction 적용.

    Returns:
        per-layer pruned count (디버깅용 — layer 이름 → 신규 prune edge 수).
    """
    counts: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, HybridGraphLinear):
            counts[name] = mod.prune_bottom_fraction(fraction)
    return counts


def _model_sparsity(model: nn.Module) -> float:
    """모든 HybridGraphLinear edge 전체에 대한 평균 sparsity."""
    total = 0
    dead = 0
    for mod in model.modules():
        if isinstance(mod, HybridGraphLinear):
            total += mod.edge_mask.numel()
            dead += int((mod.edge_mask == 0).sum().item())
    if total == 0:
        return 0.0
    return dead / total


def train_hybrid_transformer_lm(config: HybridTransformerTrainConfig) -> dict:
    """1 run 학습 — Phase 13/14/15 sweep unit.

    Phase 15: ``config.prune_at_step`` 에 도달 시 ``config.prune_fraction`` 만큼 prune.
    plain arch 는 HybridGraphLinear 가 없어 prune 무효 (HybridGraphTransformerLM 의 plain 도 동일).

    Returns: ``losses``, ``final_loss`` (last 100 mean), ``final_adj``, ``final_sparsity``,
    ``prune_event`` (prune 실행 시점의 step + 신규 prune edge 수).
    """
    set_seed(config.seed)
    model = HybridGraphTransformerLM(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        n_layers=config.n_layers,
        max_seq_len=config.block_size,
        arch=config.arch,
        group_size=config.group_size,
        dropout=config.dropout,
        use_full_graph=config.use_full_graph,
    ).to(config.device)
    data_iter = iter_random_batches(
        config.dataset, batch_size=config.batch_size, block_size=config.block_size, seed=config.seed
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    losses: list[float] = []
    prune_event: dict | None = None
    model.train()
    for step in range(1, config.max_steps + 1):
        x, y = next(data_iter)
        x, y = x.to(config.device), y.to(config.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Phase 15: prune at midpoint (or configured step)
        if (
            config.prune_at_step is not None
            and step == config.prune_at_step
            and config.prune_fraction > 0
        ):
            per_layer = _prune_model(model, config.prune_fraction)
            prune_event = {
                "step": step,
                "total_pruned": sum(per_layer.values()),
                "sparsity_after": _model_sparsity(model),
            }

    n_last = min(100, len(losses))
    final_loss = sum(losses[-n_last:]) / n_last if n_last > 0 else 0.0

    return {
        "losses": losses,
        "final_loss": final_loss,
        "final_adj": _snapshot_adj(model),
        "final_sparsity": _model_sparsity(model),
        "prune_event": prune_event,
    }


def count_parameters(model: nn.Module) -> int:
    """전체 학습 가능 파라미터 수 (arch 간 공정 비교용 reporting)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "HybridGraphTransformerLM",
    "HybridTransformerTrainConfig",
    "count_parameters",
    "train_hybrid_transformer_lm",
]
