"""Training utilities — Phase 1: triggers + simple loop."""

from graphlm.training.loop import TrainConfig, train
from graphlm.training.triggers import PlateauTrigger

__all__ = ["PlateauTrigger", "TrainConfig", "train"]
