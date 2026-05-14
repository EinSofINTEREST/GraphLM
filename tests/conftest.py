"""pytest 공통 fixture."""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fix_seed() -> None:
    """모든 테스트에 동일 시드 적용 — 재현성 보장."""
    random.seed(42)
    np.random.seed(42)
