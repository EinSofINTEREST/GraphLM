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


def _make_optimizer(model: GrowingDecoder, cfg: TrainConfig) -> torch.optim.Optimizer:
    """초기 optimizer 생성 (한 번만)."""
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _register_new_params(
    optimizer: torch.optim.Optimizer,
    new_params: list[torch.nn.Parameter],
    cfg: TrainConfig,
) -> None:
    """새 parameter 만 optimizer 에 add_param_group 으로 등록.

    Why not 재생성:
        재생성하면 기존 parameter 의 momentum (m, v) 가 모두 reset 되어 학습 spike 가능.
        본 PR 의 핵심 invariant \"function preservation 으로 spike 없음\" 과 충돌.
        add_param_group 은 기존 state 유지 + 신규 param 만 0 state 로 시작 → 안전.
    """
    optimizer.add_param_group(
        {"params": new_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}
    )


def _train_step(
    model: GrowingDecoder,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    target_ids: Tensor,
) -> float:
    """Run one forward + backward + optimizer step, return scalar loss."""
    logits = model(input_ids)  # [B, T, V]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _maybe_grow(
    model: GrowingDecoder,
    optimizer: torch.optim.Optimizer,
    trigger: PlateauTrigger,
    loss_val: float,
    cfg: TrainConfig,
    step: int,
) -> int | None:
    """If plateau detected and not at max_layers, grow + register new params.

    Returns:
        Index of newly added block, or None if no growth happened this step.
    """
    if not (trigger.update(loss_val) and model.n_layers < cfg.max_layers):
        return None
    new_idx = add_layer_function_preserving(model)
    new_params = list(model.blocks[new_idx].parameters())
    _register_new_params(optimizer, new_params, cfg)
    logger.info(
        "grow",
        extra={
            "step": step,
            "n_layers": model.n_layers,
            "new_block_idx": new_idx,
            "alpha": 0.0,
        },
    )
    return new_idx


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

    optimizer = _make_optimizer(model, cfg)
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
            logger.info("data_exhausted", extra={"step": step})
            break

        loss_val = _train_step(model, optimizer, input_ids.to(device), target_ids.to(device))
        result.losses.append(loss_val)

        if step % cfg.log_every == 0:
            logger.info(
                "train_step",
                extra={
                    "step": step,
                    "loss": loss_val,
                    "n_layers": model.n_layers,
                    "n_params": model.n_params,
                },
            )

        if _maybe_grow(model, optimizer, trigger, loss_val, cfg, step) is not None:
            result.grow_events.append((step, model.n_layers))

    result.final_n_layers = model.n_layers
    result.final_n_params = model.n_params
    return result
