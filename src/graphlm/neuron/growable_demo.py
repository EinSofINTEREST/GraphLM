"""Phase 8 — Growable MLP language model + 학습 헬퍼 (demo orchestration용).

노트북에서 함수/클래스를 직접 정의하지 말라는 `.claude/rules/06-code-style.md` 규약 준수
(Copilot #3300394761). Phase 8 노트북 18-phase8-growable-foundations.ipynb 는 여기서 import.

- ``GrowableMLPLM``: 단순 n-gram char-LM MLP. ``hidden_dim`` 이 학습 중 확장됨.
- ``make_ngram_iter``: TinyShakespeareDataset 의 sliding window n-gram → (input, target) iterator.
- ``train_growable_mlp``: 1 run 학습 (시드 / state_preserve / init_mode) → losses / widths /
  expand_events / final_loss dict 반환.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphlm.data.tinyshakespeare import TinyShakespeareDataset, iter_random_batches
from graphlm.neuron.growable import GrowableLayerNorm, GrowableLinear
from graphlm.utils import set_seed


class GrowableMLPLM(nn.Module):
    """Char n-gram MLP language model — hidden_dim 이 학습 중 확장됨.

    Architecture:
        emb (vocab × emb_dim, 고정) → flatten n_gram → fc1 (GrowableLinear) → GELU
        → ln (GrowableLayerNorm) → fc2 (GrowableLinear) → vocab logits
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, n_gram: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = GrowableLinear(emb_dim * n_gram, hidden_dim)
        self.ln = GrowableLayerNorm(hidden_dim)
        self.fc2 = GrowableLinear(hidden_dim, vocab_size)
        self.n_gram = n_gram

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_gram) → emb (B, n_gram, emb_dim) → flatten (B, n_gram*emb_dim)
        h = self.emb(x).reshape(x.shape[0], -1)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.ln(h)
        return self.fc2(h)

    def expand_hidden(
        self,
        delta: int,
        *,
        optimizer: torch.optim.Optimizer | None,
        init: str = "zero",
    ) -> None:
        """hidden_dim 을 delta 만큼 확장: fc1.expand_out → ln.expand → fc2.expand_in.

        fc1/fc2 선형부는 zero-init 시 function-preserving 이나, LN 의 mean/var 가 새 dim 입력
        값에 영향을 받아 전체 forward 는 엄밀히 동일하지 않을 수 있음 (Copilot #3300394778).
        """
        self.fc1.expand_out(delta, optimizer=optimizer, init=init)
        self.ln.expand(delta, optimizer=optimizer)
        self.fc2.expand_in(delta, optimizer=optimizer, init=init)


def make_ngram_iter(
    dataset: TinyShakespeareDataset, batch_size: int, n_gram: int, *, seed: int
) -> Iterator[tuple[Tensor, Tensor]]:
    """BLOCK_SIZE = n_gram + 1 sliding window → (input n_gram tokens, target 1 token) iterator."""
    raw = iter_random_batches(dataset, batch_size=batch_size, block_size=n_gram + 1, seed=seed)
    for x, _y in raw:
        yield x[:, :n_gram], x[:, n_gram]


def train_growable_mlp(
    *,
    dataset: TinyShakespeareDataset,
    vocab_size: int,
    seed: int,
    state_preserve: bool,
    init_mode: str,
    emb_dim: int,
    init_hidden: int,
    expand_delta: int,
    expand_steps: list[int],
    n_gram: int,
    batch_size: int,
    lr: float,
    max_steps: int,
    device: str = "cpu",
) -> dict:
    """1 run 학습 — Phase 8 sweep 의 단위.

    Returns:
        dict with keys ``losses`` (list[float]), ``widths`` (list[int]), ``expand_events``
        (list[(step, new_hidden)]), ``final_loss`` (last 100 avg).
    """
    set_seed(seed)
    model = GrowableMLPLM(vocab_size, emb_dim, init_hidden, n_gram).to(device)
    data_iter = make_ngram_iter(dataset, batch_size, n_gram, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    widths: list[int] = []
    expand_events: list[tuple[int, int]] = []
    model.train()
    for step in range(1, max_steps + 1):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        widths.append(model.fc1.out_features)

        if step in expand_steps:
            opt_arg = optimizer if state_preserve else None
            model.expand_hidden(expand_delta, optimizer=opt_arg, init=init_mode)
            expand_events.append((step, model.fc1.out_features))
            if not state_preserve:
                # state reset = AdamW state 손실 효과
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_last = min(100, len(losses))
    return {
        "losses": losses,
        "widths": widths,
        "expand_events": expand_events,
        "final_loss": sum(losses[-n_last:]) / n_last if n_last > 0 else 0.0,
    }
