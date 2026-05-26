"""Phase 12 — HybridGraphLinear demo MLP-LM + 학습 헬퍼.

노트북 분리 규약 준수. Phase 12 노트북 11-phase12-hybrid-graph-foundations.ipynb 에서 import.

4 가지 architecture 비교 (모두 0-init 금지 + magnitude rule 자동 적용):
- ``"plain"`` — 표준 nn.Linear (baseline)
- ``"hybrid_full_full"`` — outer=full + inner=full (function preserving 시작)
- ``"hybrid_identity_full"`` — outer=identity + inner=full (Phase 9 group_identity 의 hybrid 표현)
- ``"hybrid_full_around_one"`` — outer=full + inner=uniform_around_one (Phase 11 channel 의 hybrid 표현)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.data.tinyshakespeare import TinyShakespeareDataset, iter_random_batches
from graphlm.neuron.graph_hybrid import HybridGraphLinear
from graphlm.utils import set_seed

Arch = Literal[
    "plain",
    "hybrid_full_full",
    "hybrid_identity_full",
    "hybrid_full_around_one",
]


class HybridGraphMLPLM(nn.Module):
    """Hybrid graph MLP-LM — Phase 9/10/11 의 통합 데모 모델."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        n_gram: int,
        arch: Arch,
        group_size: int = 16,
    ):
        super().__init__()
        self.arch = arch
        self.n_gram = n_gram
        self.emb = nn.Embedding(vocab_size, emb_dim)
        in_f = emb_dim * n_gram
        self.fc1 = _make_linear(in_f, hidden_dim, arch, group_size)
        self.ln = nn.LayerNorm(hidden_dim)
        # fc2 는 hidden_dim → vocab_size — 직사각형 가능. identity outer 는 정방 필요라
        # fc2 는 arch 와 무관하게 hybrid_full_full 또는 plain 사용 (비교 공정성)
        fc2_arch: Arch = "plain" if arch == "plain" else "hybrid_full_full"
        self.fc2 = _make_linear(hidden_dim, vocab_size, fc2_arch, group_size)

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x).reshape(x.shape[0], -1)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.ln(h)
        return self.fc2(h)


def _make_linear(in_f: int, out_f: int, arch: Arch, group_size: int) -> nn.Module:
    if arch == "plain":
        return nn.Linear(in_f, out_f)
    if in_f % group_size != 0 or out_f % group_size != 0:
        raise ValueError(
            f"HybridGraphMLPLM 의 in_f({in_f}) / out_f({out_f}) 는 group_size({group_size}) 의 배수여야 함"
        )
    if arch == "hybrid_full_full":
        return HybridGraphLinear(
            in_f,
            out_f,
            group_size=group_size,
            adj_outer_init="full",
            adj_inner_init="full",
        )
    if arch == "hybrid_identity_full":
        return HybridGraphLinear(
            in_f,
            out_f,
            group_size=group_size,
            adj_outer_init="identity",
            adj_inner_init="full",
        )
    if arch == "hybrid_full_around_one":
        return HybridGraphLinear(
            in_f,
            out_f,
            group_size=group_size,
            adj_outer_init="full",
            adj_inner_init="uniform_around_one",
        )
    raise ValueError(f"unknown arch: {arch}")


def make_ngram_iter(
    dataset: TinyShakespeareDataset, batch_size: int, n_gram: int, *, seed: int
) -> Iterator[tuple[Tensor, Tensor]]:
    raw = iter_random_batches(dataset, batch_size=batch_size, block_size=n_gram + 1, seed=seed)
    for x, _y in raw:
        yield x[:, :n_gram], x[:, n_gram]


def train_hybrid_graph_mlp(
    *,
    dataset: TinyShakespeareDataset,
    vocab_size: int,
    seed: int,
    arch: Arch,
    emb_dim: int,
    hidden_dim: int,
    group_size: int,
    n_gram: int,
    batch_size: int,
    lr: float,
    max_steps: int,
    device: str = "cpu",
) -> dict:
    """1 run 학습 — Phase 12 sweep unit.

    Returns: ``losses``, ``final_loss``, ``final_adj`` (hybrid_* 에 한해 fc1 의 outer/inner 둘 다 snapshot).
    """
    set_seed(seed)
    model = HybridGraphMLPLM(vocab_size, emb_dim, hidden_dim, n_gram, arch, group_size).to(device)
    data_iter = make_ngram_iter(dataset, batch_size, n_gram, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _step in range(1, max_steps + 1):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    n_last = min(100, len(losses))
    final_loss = sum(losses[-n_last:]) / n_last if n_last > 0 else 0.0

    final_adj = None
    if arch != "plain":
        # graph_group_demo / graph_channel_demo 와 동일한 {"fc1": ..., "fc2": ...} 계층 구조 —
        # 공통 후처리/시각화 재사용 가능 (Copilot #3302306899). hybrid 는 fc1/fc2 각자 outer/inner.
        final_adj = {
            "fc1": {
                "outer": model.fc1.adj_outer.detach().cpu().clone(),
                "inner": model.fc1.adj_inner.detach().cpu().clone(),
            },
            "fc2": {
                "outer": model.fc2.adj_outer.detach().cpu().clone(),
                "inner": model.fc2.adj_inner.detach().cpu().clone(),
            },
        }
    return {
        "losses": losses,
        "final_loss": final_loss,
        "final_adj": final_adj,
    }
