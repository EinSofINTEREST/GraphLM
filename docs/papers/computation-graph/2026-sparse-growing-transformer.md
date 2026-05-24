---
title: "Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping"
authors: "(arXiv preprint; 저자 정보 추후 확정)"
year: 2026
venue: "arXiv preprint"
url: "https://arxiv.org/abs/2603.23998"
arxiv_id: "2603.23998"
tags: ["computation-graph", "dynamic-param", "growing", "transformer", "training-time", "sparse-depth", "attention-looping"]
status: "draft"
cited_in: []
---

# Sparse Growing Transformer — 학습 중 sparse depth allocation

## TL;DR (3줄)

- Transformer 의 **depth 를 학습 중 sparse 하게 할당** 하는 방법. "Progressive Attention Looping" 으로 동일 attention block 을 여러 번 재사용하면서 token / layer 별 sparse 한 depth budget 할당.
- 학습 시작 시점에는 작은 depth, 진행에 따라 progressive 하게 depth allocation 확장.
- Transformer 의 동적 depth 차원에서 Phase 1 (block-level growing) 과 유사한 family.

## 핵심 기여

- **Training-time sparse depth allocation** — inference-time efficient depth (e.g. early exit, MoD) 와 달리 학습 중 sparse depth 결정.
- **Progressive attention looping** — 동일 attention weights 를 여러 step 반복 적용 (loop) 함으로써 effective depth 를 늘리되 parameter 는 보존.
- **Sparse 의미** — 모든 token / 모든 step 이 같은 depth 를 거치는 게 아니라 학습된 schedule 에 따라 sparse 활성화.

## 방법 요약

- 데이터: LM pretraining (구체 dataset 은 본 paper 참조)
- 모델: Transformer + attention looping mechanism
- 학습:
  - Initial: 작은 depth (few blocks)
  - Progressive: looping count 를 단계적으로 늘림 → effective depth 확장
  - Sparse mask 가 token / position 별로 looping 적용 여부 결정
- 핵심 차이 vs Phase 1: **동일 weight 의 재사용** (RNN-like) → parameter 효율적이지만 expressive power 제한.

## 실험 / 결과

- LM benchmarks (perplexity).
- baseline transformer 대비 동일 parameter 로 더 효율적 학습.
- inference 시 sparse depth 가 자동 적용되어 latency 절감.

## 한계 / 비판적 시각

- **Weight reuse (loop)** — 진정한 depth 확장이 아니라 같은 block 의 반복. parameter 수 자체는 안 늘어남 → 본 프로젝트의 "training-time dynamic parameter count" 패러다임과 약간 다른 방향 (parameter 고정, depth 만 동적).
- token-level routing 의 sparse 결정이 routing 학습 비용 추가.
- 매우 깊은 looping count 시 vanishing/exploding 우려.

## 본 프로젝트 시사점

> **Phase 1 의 block-level dynamic depth 와 가장 직접 비교 가능한 Transformer paper**.

- **적용 가능 부분**:
  - Progressive depth allocation 의 scheduling 아이디어 → Phase 1 의 PlateauTrigger 가 trigger 시점만 결정하는 것보다 더 정교한 schedule 가능
  - sparse 활성화 mask → Phase 4 (Top-KAST) 와 결합 시 depth + sparsity 의 2-axis dynamic
- **차용할 아이디어**:
  - 학습 초반 → 후반의 depth growth schedule (immediate full vs progressive) 의 비교 실험 가치
- **채택하지 않을 부분**:
  - **Weight reuse (loop)** — 본 프로젝트의 패러다임은 "parameter 수 변동". loop 는 parameter 고정. 직접 채택 X.
- **후속 실험 가설**:
  - Sparse Growing Transformer 의 final loss 가 Phase 1 의 baseline 과 어떻게 비교되는가? (parameter 효율성 vs capacity 효율성 trade-off)
  - looping 과 actual block addition 의 hybrid 가 dead block 문제를 우회하는가?

## 참고 / 인용

- 관련 논문: [Universal Transformer](2018-universal-transformer-dehghani.md) (weight reuse 원형), [MoD (Mixture of Depths)](../computation-graph/2024-mod-raposo.md) (sparse depth 의 inference 버전), [LiGO](2023-ligo-wang.md) (Phase 2 의 width growth baseline)
- 본 프로젝트 내 인용 위치: Phase 1 와 직접 비교 reference
