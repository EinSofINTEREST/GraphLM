"""Phase 9 — GroupGraphLinear demo MLP-LM + 학습 헬퍼.

노트북 분리 규약 준수 (.claude/rules/06-code-style.md). Phase 9 노트북
08-phase9-group-graph-foundations.ipynb 는 여기서 import.

세 가지 architecture 를 같은 학습 루프로 비교:
- ``"plain"`` — 표준 nn.Linear (baseline)
- ``"group_full"`` — GroupGraphLinear with adj_init="full" (function preserving 시작)
- ``"group_identity"`` — GroupGraphLinear with adj_init="identity" (block-diagonal, 가장 sparse)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.data.tinyshakespeare import TinyShakespeareDataset, iter_random_batches
from graphlm.neuron.graph_group import GroupGraphLinear
from graphlm.utils import set_seed

Arch = Literal["plain", "group_full", "group_identity"]


class GroupGraphMLPLM(nn.Module):
    """Phase 8 의 GrowableMLPLM 의 graph 버전 — fc1 / fc2 가 architecture 별로 다름.

    Architecture:
        emb (vocab × emb_dim) → flatten n_gram → fc1 → GELU → LayerNorm → fc2 → vocab logits
    """

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
        # fc2 는 hidden_dim → vocab_size (직사각형 — n_groups_out ≠ n_groups_in 가능).
        # arch="group_identity" 의 의미는 정방에서만 정의되므로 fc2 는 항상 "full" 사용 —
        # 비교 공정성 유지 (group_identity 비교의 차이는 fc1 에 한정) (Copilot #3301531369).
        fc2_arch: Arch = "plain" if arch == "plain" else "group_full"
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
    # group_*: in_f / out_f 가 group_size 의 배수가 아니면 명시적으로 ValueError —
    # padding 은 caller (예: 노트북 V_PADDED) 책임이며 본 함수는 검증만 (Copilot #3301531359).
    if in_f % group_size != 0 or out_f % group_size != 0:
        raise ValueError(
            f"GroupGraphMLPLM 의 in_f({in_f}) / out_f({out_f}) 는 group_size({group_size}) 의 배수여야 함"
        )
    adj_init = "full" if arch == "group_full" else "identity"
    return GroupGraphLinear(in_f, out_f, group_size=group_size, adj_init=adj_init)


def make_ngram_iter(
    dataset: TinyShakespeareDataset, batch_size: int, n_gram: int, *, seed: int
) -> Iterator[tuple[Tensor, Tensor]]:
    raw = iter_random_batches(dataset, batch_size=batch_size, block_size=n_gram + 1, seed=seed)
    for x, _y in raw:
        yield x[:, :n_gram], x[:, n_gram]


def train_group_graph_mlp(
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
    """1 run 학습 — Phase 9 sweep 의 단위.

    Returns dict with ``losses``, ``final_loss`` (last 100 avg), ``final_adj`` (학습된 adjacency
    snapshots — plain 의 경우 None).
    """
    set_seed(seed)
    model = GroupGraphMLPLM(vocab_size, emb_dim, hidden_dim, n_gram, arch, group_size).to(device)
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

    # adjacency snapshot (group_* 에만 있음)
    final_adj = None
    if arch != "plain":
        final_adj = {
            "fc1": model.fc1.adj.detach().cpu().clone(),
            "fc2": model.fc2.adj.detach().cpu().clone(),
        }
    return {
        "losses": losses,
        "final_loss": final_loss,
        "final_adj": final_adj,
    }
