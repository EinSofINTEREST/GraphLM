---
title: "Differentiable plasticity: training plastic neural networks with backpropagation"
authors: "Miconi, T., Stanley, K. O., & Clune, J."
year: 2018
venue: "ICML 2018"
url: "https://arxiv.org/abs/1804.02464"
arxiv_id: "1804.02464"
tags: ["computation-graph", "plasticity", "differentiable", "fast-weights", "online-learning", "structural-foundation"]
status: "draft"
cited_in: []
---

# Differentiable Plasticity — Plasticity Rule 자체를 학습

## TL;DR (3줄)

- 신경망의 **fast weights** (학습 중 빠르게 변하는 추가 weight) 와 그 weight 의 **plasticity coefficient** 를 동시에 backprop 으로 학습.
- 한 connection 의 weight = $w_{ij} + \alpha_{ij} \cdot \text{Hebb}_{ij}$ — slow weight $w$, plasticity 계수 $\alpha$, Hebbian trace $\text{Hebb}$ 의 합성.
- meta-learning 의 outer loop 에서 $w$ 와 $\alpha$ 모두 학습 → inner loop 에서 $\text{Hebb}$ 가 빠르게 적응 → 작은 episode 에서 빠른 학습.

## 핵심 기여

- **Plasticity 계수 $\alpha$ 의 differentiability** — 기존 Hebbian 학습은 fixed rule, 본 paper 는 $\alpha$ 자체를 학습.
- **Slow + fast weight 의 분리** — slow $w$ 는 backprop 으로 outer learning, fast $\text{Hebb}$ 는 episode 안에서 빠른 plasticity.
- **Memorization / pattern completion** task 에서 baseline RNN/LSTM 대비 큰 폭 우위.

## 방법 요약

- 데이터/task: image reconstruction, omniglot one-shot learning, RL maze.
- 모델:
  - Recurrent network with plastic weights
  - 각 connection: $w_{ij} + \alpha_{ij} \cdot \text{Hebb}_{ij}(t)$ (원논문 §2 Eq. 1)
  - $\text{Hebb}_{ij}(t+1) = \eta \cdot x_i(t) \cdot y_j(t) + (1-\eta) \cdot \text{Hebb}_{ij}(t)$ (Hebbian trace, 원논문 §2 Eq. 2)
- 학습:
  - Outer (meta): backprop 으로 $w$, $\alpha$, $\eta$ 학습
  - Inner (episode): $\text{Hebb}$ 만 빠르게 변동
- 핵심: $\alpha$ 가 0 이면 connection 은 plasticity 없음 (slow weight 만), $\alpha$ 가 크면 빠르게 적응.

## 실험 / 결과

- Omniglot one-shot: standard RNN/LSTM 대비 큰 폭 accuracy 향상.
- RL maze: 환경 변화에 빠른 적응.
- 재현성: 공식 PyTorch / TF 구현 공개.

## 한계 / 비판적 시각

- **Structural change 아님** — connection 자체의 추가/제거는 X. weight value 의 빠른 변화만.
- $\alpha$ 가 connection 별 — large-scale Transformer 에 직접 적용 시 parameter overhead 2x.
- meta-learning 의 outer loop 비용.

## 본 프로젝트 시사점

> **본 프로젝트의 α (residual scale) 가 "plasticity coefficient" 와 본질적으로 같은 아이디어**.

- **적용 가능 부분**:
  - Phase 1 의 α (block 당 1 scalar) → Differentiable Plasticity 의 α (connection 당 1 scalar) 의 확장형
  - α 가 0 이면 dead — 정확히 본 paper 의 plasticity=0 = no fast adaptation 과 같음
- **차용할 아이디어**:
  - **Slow + fast weight 분리** — Phase 1 의 weight 와 α 를 명시적으로 두 timescale 로 분리하면 dead block 회피 가능?
  - **Connection-level α** — 본 프로젝트의 block-level α 보다 fine-grained. 사용자의 function-level dynamic 컨셉의 직접 reference.
  - **Hebbian trace** — gradient 없이 local 신호로 weight update — Phase 1 의 dead block 의 weight gradient ≈ 0 문제를 우회 가능
- **채택하지 않을 부분**:
  - Recurrent network 구조
  - Meta-learning 의 outer/inner loop — 본 프로젝트는 standard pretraining
- **후속 실험 가설**:
  - Phase 1 의 α 를 connection-level 로 확장 (각 sublayer 안의 weight 매트릭스 entry 마다 α) — function-level dynamic 의 한 구현
  - Hebbian trace 를 추가 block 의 init 신호로 사용 (random init 대신) — dead block 회피
  - α 와 weight 를 다른 lr 로 학습 (α 빠르게, w 느리게) — 자연스러운 plasticity dynamics

## 참고 / 인용

- 공식 코드: <https://github.com/uber-research/differentiable-plasticity> (PyTorch / TF)
- 관련 논문: [SMGrNN](2025-smgrnn.md) (structural plasticity 확장), [SpikePropamine](https://pmc.ncbi.nlm.nih.gov/articles/PMC8493296/) (spiking 버전)
- 본 프로젝트 내 인용 위치: α 의 개념적 foundation, function-level dynamic 의 plasticity 차원 reference
