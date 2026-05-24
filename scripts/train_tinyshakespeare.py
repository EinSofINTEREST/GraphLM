#!/usr/bin/env python3
"""Phase 1 검증 entry point — TinyShakespeare 로 growing Transformer 학습.

핵심 검증:
1. 학습이 정상 수렴하는가
2. PlateauTrigger 가 실제로 fire 하는가
3. fire 직후 function preservation 으로 spike 없는가
4. 학습 끝 시점에 더 깊은 모델이 된 채로 더 낮은 loss 도달하는가

사용법:
  python3 scripts/train_tinyshakespeare.py [--max-steps 5000] [--device cpu]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from graphlm.data.tinyshakespeare import (
    CharTokenizer,
    TinyShakespeareDataset,
    iter_random_batches,
    load_tinyshakespeare_text,
)
from graphlm.models.backbone import BackboneConfig, GrowingDecoder
from graphlm.training.loop import TrainConfig, train
from graphlm.utils.seed import set_seed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-layers", type=int, default=8)
    ap.add_argument("--trigger-window", type=int, default=200)
    ap.add_argument("--trigger-epsilon", type=float, default=0.05)
    ap.add_argument("--trigger-cooldown", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-path", default="data/tinyshakespeare.txt")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    set_seed(args.seed)

    log.info("loading TinyShakespeare …")
    text = load_tinyshakespeare_text(args.data_path)
    log.info("text length=%d chars", len(text))

    tokenizer = CharTokenizer(text)
    log.info("vocab_size=%d", tokenizer.vocab_size)

    dataset = TinyShakespeareDataset(text, tokenizer)
    data_iter = iter_random_batches(
        dataset,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed,
    )

    cfg = BackboneConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        n_heads=4,
        ffn_dim=1024,
        max_seq_len=args.block_size,
        n_init_layers=4,
    )
    model = GrowingDecoder(cfg)
    log.info("init model: n_layers=%d n_params=%d", model.n_layers, model.n_params)

    train_cfg = TrainConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        max_layers=args.max_layers,
        trigger_window=args.trigger_window,
        trigger_epsilon=args.trigger_epsilon,
        trigger_cooldown=args.trigger_cooldown,
        trigger_min_history=args.trigger_window,
        device=args.device,
    )
    result = train(model, data_iter, train_cfg)

    log.info("=" * 60)
    log.info("training complete")
    log.info("  final n_layers = %d", result.final_n_layers)
    log.info("  final n_params = %d", result.final_n_params)
    log.info("  grow events   = %d", len(result.grow_events))
    for step, n in result.grow_events:
        log.info("    step=%d → n_layers=%d", step, n)
    if result.losses:
        first_avg = sum(result.losses[:100]) / min(100, len(result.losses))
        last_avg = sum(result.losses[-100:]) / min(100, len(result.losses))
        log.info("  loss first 100 avg = %.4f", first_avg)
        log.info("  loss last  100 avg = %.4f", last_avg)

    # 결과 저장 (간단 CSV)
    out_dir = Path("runs/tinyshakespeare")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "losses.csv").open("w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(result.losses, 1):
            f.write(f"{i},{loss:.6f}\n")
    with (out_dir / "grow_events.csv").open("w") as f:
        f.write("step,n_layers_after\n")
        for step, n in result.grow_events:
            f.write(f"{step},{n}\n")
    log.info("saved losses to runs/tinyshakespeare/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
