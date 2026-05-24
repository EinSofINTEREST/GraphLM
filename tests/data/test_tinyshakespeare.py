"""CharTokenizer + batch iterator unit tests (no network — uses inline text)."""

from __future__ import annotations

import pytest
import torch

from graphlm.data.tinyshakespeare import (
    CharTokenizer,
    TinyShakespeareDataset,
    iter_random_batches,
)

SAMPLE = (
    "To be, or not to be, that is the question.\n"
    "Whether 'tis nobler in the mind to suffer..."
)


def test_tokenizer_roundtrip():
    tok = CharTokenizer(SAMPLE)
    ids = tok.encode("be or not")
    assert tok.decode(ids) == "be or not"


def test_tokenizer_vocab_size_matches_unique_chars():
    tok = CharTokenizer(SAMPLE)
    assert tok.vocab_size == len(set(SAMPLE))


def test_dataset_size():
    tok = CharTokenizer(SAMPLE)
    ds = TinyShakespeareDataset(SAMPLE, tok)
    assert len(ds) == len(SAMPLE)


def test_batch_iterator_shape():
    tok = CharTokenizer(SAMPLE * 100)
    ds = TinyShakespeareDataset(SAMPLE * 100, tok)
    it = iter_random_batches(ds, batch_size=4, block_size=8, n_batches=3, seed=0)
    batches = list(it)
    assert len(batches) == 3
    for x, y in batches:
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        assert x.dtype == torch.long


def test_batch_iterator_too_small_raises():
    tok = CharTokenizer("abc")
    ds = TinyShakespeareDataset("abc", tok)
    with pytest.raises(ValueError):
        next(iter_random_batches(ds, batch_size=1, block_size=10))


@pytest.mark.network
def test_load_tinyshakespeare_downloads(tmp_path):
    """Network test — downloads ~1MB."""
    from graphlm.data.tinyshakespeare import load_tinyshakespeare_text

    text = load_tinyshakespeare_text(tmp_path / "input.txt")
    assert len(text) > 1_000_000
    assert "Shakespeare" in text or "Romeo" in text
