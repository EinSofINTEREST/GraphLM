---
title: "Lifelong Learning with Dynamically Expandable Networks"
authors: "Yoon, J., Yang, E., Lee, J., & Hwang, S. J."
year: 2018
venue: "ICLR 2018"
url: "https://arxiv.org/abs/1708.01547"
arxiv_id: "1708.01547"
code_url: "https://github.com/jaehong31/DEN"
tags: ["computation-graph", "dynamic-param", "adaptive-trigger", "den", "continual-learning", "capacity-expansion", "selective-retrain"]
status: "draft"
cited_in: []
---

# DEN — Dynamically Expandable Networks (capacity 부족 시 확장)

## TL;DR (3줄)

- Continual learning 의 새 task 가 도착했을 때, **기존 capacity 로 fit 실패하면 (loss threshold 초과) 자동으로 새 neuron 추가** — adaptive trigger 기반 확장의 시초.
- 3-stage scheme: selective retraining → dynamic network expansion (필요 시) → network split/duplication (drift 방지).
- 본 프로젝트 관점의 가치: **\"성장의 trigger 가 schedule 이 아니라 학습 신호\"** 라는 패턴의 첫 명시 — Net2Net 류와의 대비.

## 핵심 기여

- **Trigger-based expansion**: 새 task 학습 후 loss 가 threshold $\tau$ 를 초과하면 → 그 task 의 hidden representation 확장 필요 → 새 neuron 추가.
- **Group-sparsity regularization** 으로 추가된 neuron 의 \"의미 있는 사용\" 강제. 의미 없는 추가는 pruning.
- **Selective retraining**: 새 task 학습 시 모든 weight 가 아니라 task-relevant subnetwork 만 갱신 → catastrophic forgetting 완화.
- **Duplication on drift**: weight 가 크게 변하면 (semantic drift) 기존 neuron 을 복제해 분리 → 이전 task 보존.

## 방법 요약

- 데이터: MNIST-permutations (각 task 가 다른 pixel permutation), CIFAR-100 (10개 task), AWA (animals with attributes).
- 모델: MLP / CNN backbone. Continual learning sequence.
- 학습 (per task $t$):
  1. **Selective retraining**: 새 task data 로 sparse 갱신 (group lasso)
  2. Trigger: validation loss $\ell^{(t)} > \tau$ 이면 expansion
  3. **Dynamic expansion**: 각 layer 마다 $k$ 개 new neuron 추가, 다시 학습 + group-sparsity 로 의미 없는 것 prune
  4. **Drift detection**: weight 변동 > threshold 면 duplicate
- 핵심 흐름:

각 layer $l$ 의 expansion 결정:
$$
\text{expand}(l, t) = \begin{cases} \text{True} & \ell^{(t)} > \tau \\ \text{False} & \text{otherwise} \end{cases}
$$

## 실험 / 결과

- MNIST-permutations 10 tasks: DEN 의 average accuracy 가 EWC / progressive 등 baseline 대비 우위 + 파라미터 효율 (capacity 가 적절히만 확장).
- CIFAR-100 / AWA 유사 패턴.
- Trigger 가 효과적으로 작동 — 모든 task 에서 균등 확장이 아니라 task difficulty 에 따라 확장 폭 자동 조절 보고.
- 재현성: 공식 TensorFlow + PyTorch 포트.

## 한계 / 비판적 시각

- Threshold $\tau$ 가 hyperparameter — task / dataset 에 따라 튜닝 부담.
- **Continual learning setting 위주** — single task 의 학습 중 확장에는 직접 적용 어려움 (본 프로젝트의 핵심 setting 과 차이).
- Trigger 가 \"task 끝난 후 validation loss\" 기반 — sub-epoch 수준의 fine-grained trigger 미해결.
- Sparse retraining + drift detection 의 결합이 hyperparameter 다수 → 학습 안정성 / 재현성 우려.

## 본 프로젝트 시사점

- **Adaptive trigger 의 reference foundation** — 본 프로젝트가 \"학습 신호로 노드 추가\" 를 구현할 때 DEN 의 \"loss > $\tau$ → expand\" 패턴이 minimum recipe.
- **차용할 아이디어**:
  - **Loss-threshold trigger** — 단순하지만 효과적. 본 프로젝트 첫 trigger 구현 시 baseline.
  - **Group-sparsity 로 의미 없는 확장 정리** — 추가했다가 학습 후 의미 없으면 자동 prune → 과확장 방지.
  - **Drift detection** — 큰 weight 변동을 모니터링하는 별도 메커니즘.
- **채택하지 않을 부분**: continual learning 의 task-sequential setting — 본 프로젝트는 single-task long pretrain 가능성 큼. trigger 자체만 차용, task boundary 가정은 버림.
- **후속 실험 가설**:
  - DEN 의 task-boundary trigger 를 **\"loss plateau 감지\"** trigger 로 일반화 (sliding window 의 loss 분산이 임계 이하면 plateau → expand) — single-task 에서도 작동하는지 검증.
  - Firefly 의 gradient-based 와 DEN 의 loss-based trigger 의 quality 비교 — 어느 신호가 더 reliable 한지.

## 참고 / 인용

- 공식 코드: <https://github.com/jaehong31/DEN> (TensorFlow)
- 관련 논문: [Firefly NAD](2020-firefly-wu.md) (gradient 기반 trigger), [AutoGrow](2020-autogrow-wen.md) (depth growth trigger), [Self-Expanding NN](2024-self-expanding-mitchell.md) (modern adaptive)
- 본 프로젝트 내 인용 위치: 추후 adaptive trigger 첫 prototype 노트북에서
