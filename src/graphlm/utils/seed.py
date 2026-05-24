"""Seed fixing for reproducibility (per 01-architecture.md)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for random / numpy / torch (including CUDA if available).

    Phase 1 의 모든 실험에서 호출 — function preservation 검증과
    학습 곡선 재현성 보장의 prerequisite.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
