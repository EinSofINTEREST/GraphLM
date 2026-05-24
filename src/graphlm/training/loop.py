"""Training loop with growth callback (Phase 1).

검증 시나리오:
1. 작은 backbone 으로 시작
2. 표준 AdamW 학습
3. 매 step 마다 PlateauTrigger.update(loss) 호출
4. trigger fire → Net2DeeperNet 으로 layer 추가
5. AdamW state 갱신 (새 parameter 등록)
6. 학습 계속
7. max_layers 도달 또는 max_steps 종료
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch import Tensor

from graphlm.growth.net2deeper import add_layer_function_preserving
from graphlm.models.backbone import GrowingDecoder
from graphlm.training.triggers import PlateauTrigger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Phase 1 학습 hyperparameter (frozen for safety)."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 5000
    log_every: int = 50
    # Growth control
    max_layers: int = 8
    trigger_window: int = 200
    trigger_epsilon: float = 0.05
    trigger_cooldown: int = 200
    trigger_min_history: int = 200
    # Device
    device: str = "cpu"


@dataclass
class TrainResult:
    """학습 결과 — loss curve + grow event timeline."""

    losses: list[float] = field(default_factory=list)
    grow_events: list[tuple[int, int]] = field(default_factory=list)  # (step, n_layers_after)
    final_n_layers: int = 0
    final_n_params: int = 0


def _rebuild_optimizer(model: GrowingDecoder, cfg: TrainConfig) -> torch.optim.Optimizer:
    """새 parameter 가 추가된 후 optimizer 를 재생성.

    AdamW 의 internal state (m, v moments) 는 기존 parameter 의 것만 갖고 있어
    그대로 두면 새 parameter 의 state 가 없음. 가장 단순한 방법은 재생성 — moment 가
    초기화되어 \"warm restart\" 효과. Phase 1 minimum 으로 충분.
    """
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def train(
    model: GrowingDecoder,
    data_iter: Iterator[tuple[Tensor, Tensor]],
    cfg: TrainConfig,
) -> TrainResult:
    """Train the model with periodic plateau-triggered growth.

    Args:
        model: GrowingDecoder instance (modified in place).
        data_iter: Iterator yielding (input_ids, target_ids) batches, both shape [B, T].
        cfg: TrainConfig.

    Returns:
        TrainResult with loss curve and grow events.
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.train()

    optimizer = _rebuild_optimizer(model, cfg)
    trigger = PlateauTrigger(
        window=cfg.trigger_window,
        epsilon=cfg.trigger_epsilon,
        cooldown=cfg.trigger_cooldown,
        min_history=cfg.trigger_min_history,
    )
    result = TrainResult()

    for step in range(1, cfg.max_steps + 1):
        try:
            input_ids, target_ids = next(data_iter)
        except StopIteration:
            logger.info("data iterator exhausted at step %d", step)
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # [B, T, V]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        result.losses.append(loss_val)

        # 학습 진행 로그
        if step % cfg.log_every == 0:
            logger.info(
                "step=%d loss=%.4f n_layers=%d n_params=%d",
                step,
                loss_val,
                model.n_layers,
                model.n_params,
            )

        # Trigger check
        if trigger.update(loss_val) and model.n_layers < cfg.max_layers:
            new_idx = add_layer_function_preserving(model)
            optimizer = _rebuild_optimizer(model, cfg)
            result.grow_events.append((step, model.n_layers))
            logger.info(
                "🌱 GROW @ step=%d → n_layers=%d (new block idx=%d, alpha=0)",
                step,
                model.n_layers,
                new_idx,
            )

    result.final_n_layers = model.n_layers
    result.final_n_params = model.n_params
    return result
