"""Phase 10 — ChannelGraphLinear demo MLP-LM + 학습 헬퍼.

노트북 분리 규약 준수. Phase 10 노트북 09-phase10-channel-graph-foundations.ipynb 는 여기서
import. Phase 9 의 ``graph_group_demo`` 의 channel-level 대응.

3 가지 architecture 를 같은 학습 루프로 비교:
- ``"plain"`` — 표준 nn.Linear (baseline)
- ``"channel_full"`` — ChannelGraphLinear with adj_init="full" (function preserving 시작)
- ``"channel_uniform_small"`` — ChannelGraphLinear with adj_init="uniform_small"
  (Phase 2 sweet spot 패턴 channel-level edge 에 적용)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.data.tinyshakespeare import TinyShakespeareDataset, iter_random_batches
from graphlm.neuron.graph_channel import ChannelGraphLinear
from graphlm.utils import set_seed

Arch = Literal["plain", "channel_full", "channel_uniform_small"]


class ChannelGraphMLPLM(nn.Module):
    """Channel-as-node graph MLP-LM — Phase 9 의 GroupGraphMLPLM 의 channel-level 버전."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        n_gram: int,
        arch: Arch,
    ):
        super().__init__()
        self.arch = arch
        self.n_gram = n_gram
        self.emb = nn.Embedding(vocab_size, emb_dim)
        in_f = emb_dim * n_gram
        self.fc1 = _make_linear(in_f, hidden_dim, arch)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = _make_linear(hidden_dim, vocab_size, arch)

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x).reshape(x.shape[0], -1)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.ln(h)
        return self.fc2(h)


def _make_linear(in_f: int, out_f: int, arch: Arch) -> nn.Module:
    if arch == "plain":
        return nn.Linear(in_f, out_f)
    if arch == "channel_full":
        return ChannelGraphLinear(in_f, out_f, adj_init="full")
    if arch == "channel_uniform_small":
        return ChannelGraphLinear(in_f, out_f, adj_init="uniform_small")
    raise ValueError(f"unknown arch: {arch}")


def make_ngram_iter(
    dataset: TinyShakespeareDataset, batch_size: int, n_gram: int, *, seed: int
) -> Iterator[tuple[Tensor, Tensor]]:
    raw = iter_random_batches(dataset, batch_size=batch_size, block_size=n_gram + 1, seed=seed)
    for x, _y in raw:
        yield x[:, :n_gram], x[:, n_gram]


def train_channel_graph_mlp(
    *,
    dataset: TinyShakespeareDataset,
    vocab_size: int,
    seed: int,
    arch: Arch,
    emb_dim: int,
    hidden_dim: int,
    n_gram: int,
    batch_size: int,
    lr: float,
    max_steps: int,
    device: str = "cpu",
) -> dict:
    """1 run 학습 — Phase 10 sweep 의 단위.

    Returns dict: ``losses``, ``final_loss``, ``final_adj`` (channel_* 의 경우 fc1/fc2 의 adj snapshot).
    """
    set_seed(seed)
    model = ChannelGraphMLPLM(vocab_size, emb_dim, hidden_dim, n_gram, arch).to(device)
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
        final_adj = {
            "fc1": model.fc1.adj.detach().cpu().clone(),
            "fc2": model.fc2.adj.detach().cpu().clone(),
        }
    return {
        "losses": losses,
        "final_loss": final_loss,
        "final_adj": final_adj,
    }
