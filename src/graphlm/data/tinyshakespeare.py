"""TinyShakespeare char-LM dataset.

Phase 1 검증용 toy dataset — 약 1MB 텍스트, char-level tokenization.
- Vocabulary: 65 character (Shakespeare 의 unique char 집합)
- Sequence: rolling window of fixed block_size

다운로드 URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_tinyshakespeare_text(cache_path: Path | str = "data/tinyshakespeare.txt") -> str:
    """Download (once) and return the TinyShakespeare text.

    Caches under cache_path (default ``data/tinyshakespeare.txt`` — gitignored).
    """
    path = Path(cache_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Minimal char-level tokenizer (no special tokens, no BPE)."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.itos: list[str] = chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        """Encode a string into a list of token ids (unknown chars are silently dropped)."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string (out-of-range ids are dropped)."""
        return "".join(self.itos[i] for i in ids if 0 <= i < self.vocab_size)


class TinyShakespeareDataset:
    """Holds encoded token tensor + provides random-batch sampling."""

    def __init__(self, text: str, tokenizer: CharTokenizer):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return self.data.shape[0]


def iter_random_batches(
    dataset: TinyShakespeareDataset,
    *,
    batch_size: int,
    block_size: int,
    n_batches: int | None = None,
    seed: int = 0,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Random-offset rolling-window batch iterator.

    각 sample 은 길이 ``block_size`` 의 (input, target) pair — target 은 input 의 1-step shift.

    Args:
        dataset: TinyShakespeareDataset.
        batch_size: B
        block_size: T
        n_batches: 총 batch 수. None 이면 무한 iteration.
        seed: torch.Generator seed.

    Yields:
        (input_ids [B, T], target_ids [B, T])
    """
    g = torch.Generator()
    g.manual_seed(seed)
    n = len(dataset)
    if n <= block_size:
        raise ValueError(f"dataset too small: {n} <= block_size {block_size}")
    count = 0
    while n_batches is None or count < n_batches:
        # randint high 는 exclusive. i 의 max = n - block_size - 1 (inclusive) → high = n - block_size
        ix = torch.randint(0, n - block_size, (batch_size,), generator=g)
        x = torch.stack([dataset.data[i : i + block_size] for i in ix])
        y = torch.stack([dataset.data[i + 1 : i + 1 + block_size] for i in ix])
        yield x, y
        count += 1
