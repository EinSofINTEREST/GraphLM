---
title: "Modular Networks: Learning to Decompose Neural Computation"
authors: "Kirsch, L., Kunze, J., & Barber, D."
year: 2018
venue: "NeurIPS 2018"
url: "https://arxiv.org/abs/1811.05249"
arxiv_id: "1811.05249"
tags: ["computation-graph", "modular", "decomposition", "routing", "differentiable", "expert-selection", "foundation"]
status: "draft"
cited_in: []
---

# Modular Networks — 학습 가능한 Computation Decomposition

## TL;DR (3줄)

- **모듈 (sub-network)** 와 **routing controller** 를 동시 학습 — 입력에 따라 어느 모듈을 활성화할지 controller 가 결정.
- Controller 의 routing 이 hard (discrete) 하면서도 학습 가능 — Gumbel-Softmax / REINFORCE 변형으로 backprop.
- 학습이 progress 함에 따라 모듈이 specialized (e.g. arithmetic operator 별 specialization) — emergent decomposition.

## 핵심 기여

- **Hard routing + 학습 가능** — soft (weighted sum) 가 아닌 hard (top-1 module 선택) routing 을 differentiable 로 학습.
- **Module specialization emergence** — 학습 중 모듈이 자연스럽게 다른 sub-task 에 특화 (interpretable).
- **Conditional computation** — 매 input 마다 다른 모듈 활성화 → parameter efficiency.

## 방법 요약

- 데이터: MNIST, CIFAR-10, language task (arithmetic).
- 모델: $K$ 개의 module + controller (small network) + routing.
- 학습:
  - Forward: controller 가 module index 결정 (Gumbel-Softmax)
  - 활성 module 만 forward
  - Backward: gradient 가 활성 module + controller 로 흐름
- 핵심: hard routing 의 discrete decision 이 backprop 통과 가능 (Gumbel trick).

## 실험 / 결과

- MNIST classification: 모듈이 digit class 별 specialized.
- Arithmetic task: 모듈이 +/-/× 등 operator 별 specialized (interpretable decomposition).
- Parameter 효율 + interpretability 둘 다 baseline 대비 우위.

## 한계 / 비판적 시각

- **모듈 수 K 가 hyperparameter** — 자동 결정 X.
- Hard routing 의 학습 stability — Gumbel-Softmax 의 temperature schedule 민감.
- 모든 모듈을 메모리에 보유 (각 input 은 1개만 사용해도) → memory 효율은 낮음.

## 본 프로젝트 시사점

> **본 프로젝트의 "동적 synapse + 동적 노드" 의 routing 차원의 foundation**.

- **적용 가능 부분**:
  - Controller 의 routing → 사용자 컨셉의 "동적 synapse" 의 직접 구현
  - 학습 가능 hard routing → function-level 노드 중 어느 것을 활성화할지 학습 가능
- **차용할 아이디어**:
  - **Gumbel-Softmax routing** — sigmoid 보다 더 정교한 differentiable hard selection. 사용자 컨셉의 동적 synapse 의 1차 후보 메커니즘
  - **Emergent specialization** — Phase 1 의 추가 block 이 dead 인 이유 중 하나가 "특화할 task 가 없어서" 일 수 있음. routing 으로 인위적으로 specialization 유도 가능
- **채택하지 않을 부분**:
  - K 모듈 메모리 보유 — 본 프로젝트는 sequential block. parallel module 은 차원 다름
- **후속 실험 가설**:
  - Phase 1 의 모든 block 을 sequential 대신 parallel + routing 으로 바꾸면 dead block 회피 가능한가?
  - Gumbel-Softmax routing 의 temperature 가 학습 후반에 낮아질수록 (hard 화) 모듈 specialization 강해지는가?
  - Function-level (qkv vs fc1 vs ...) 노드 중 어느 단위에서 routing 이 가장 효과적인가?

## 참고 / 인용

- 공식 코드: <https://github.com/sklbancor/Modular-Networks> (관련 implementations)
- 관련 논문: [Modular Deep Learning](2023-modular-deep-learning-pfeiffer.md) (survey), [MoE](2017-moe-shazeer.md) (parallel module routing), [Are Neural Nets Modular?](2021-modular-transformer-csordas.md) (post-hoc 분석)
- 본 프로젝트 내 인용 위치: 사용자 컨셉의 dynamic synapse 메커니즘 foundation
