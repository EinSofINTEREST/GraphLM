---
title: "Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks"
authors: "Csordás, R., van Steenkiste, S., & Schmidhuber, J."
year: 2021
venue: "ICLR 2021"
url: "https://arxiv.org/abs/2010.02066"
arxiv_id: "2010.02066"
tags: ["computation-graph", "modular", "transformer", "weight-mask", "differentiable", "interpretability", "function-level"]
status: "draft"
cited_in: []
---

# Are Neural Nets Modular? — Differentiable Weight Mask 로 Transformer 의 Functional Modularity 분석

## TL;DR (3줄)

- 학습된 신경망 (RNN / Transformer / FFN / CNN) 에 **binary weight mask** 를 학습으로 적용해 특정 기능을 담당하는 sub-network (circuit) 식별.
- "Function 단위 modularity" 가 emergence 하는지를 정량 측정 — Transformer 의 head / layer 가 task 별 specialization 을 보이는지.
- 결과: **부분적 functional modularity 존재** 하지만 완전한 modular decomposition 은 아님. weight overlap 이 상당.

## 핵심 기여

- **Differentiable weight mask** — sigmoid 의 hard threshold 로 binary mask 학습. mask 는 weight 별 on/off.
- **Task-specific subnetwork identification** — task A 와 task B 에 대해 독립 mask 학습 → overlap 측정 = modularity 정도.
- **Transformer 특화 분석** — head / FFN / embedding 단위로 modularity 측정. attention head 가 가장 modular.

## 방법 요약

- 데이터: SCAN, COGS, addition (compositional generalization).
- 모델: Transformer, LSTM, FFN.
- 학습:
  - Pretrained model 에 weight mask 학습 (mask 만 update, weight 고정)
  - mask 가 sparse + task-specific 하도록 regularize
  - mask 의 overlap 측정 → modularity score
- 핵심: mask 가 task A 와 B 에서 거의 disjoint → modular, 완전 overlap → not modular.

## 실험 / 결과

- Transformer 의 **attention head 가 가장 modular** (task 별 distinct subnetwork).
- FFN 은 부분적으로 shared.
- LSTM 은 거의 non-modular (모든 weight 가 모든 task 에 관여).
- compositional generalization 성능과 modularity score 가 상관.

## 한계 / 비판적 시각

- **Post-hoc analysis** — 학습된 모델에 mask 적용. 학습 중 modular structure 를 induce 하는 방법은 X.
- Mask 학습 자체의 비용.
- "Modular" 의 정의가 weight overlap 만으로 측정 — semantic modular 와 다를 수 있음.

## 본 프로젝트 시사점

> **본 프로젝트의 function-level 노드 컨셉이 사후 분석으로 자연스럽게 emerge 하는 것을 입증하는 paper**.

- **적용 가능 부분**:
  - Phase 1 의 학습된 모델 (특히 dead block 들) 에 differentiable mask 적용 → 어느 weight 가 정말 dead 인지 정량 측정 가능
  - Transformer 의 attention head 가 가장 modular 라는 발견 → 사용자 컨셉의 "함수 단위 노드" 중 attention head 가 가장 유망한 단위
- **차용할 아이디어**:
  - **Differentiable binary mask** — sigmoid + hard threshold 패턴. 사용자 컨셉의 "동적 synapse" 의 differentiable 구현 직접 후보
  - **Overlap 기반 modularity score** — Phase 1 의 추가 block 이 초기 block 과 functional overlap 이 얼마나 되는지 정량화
- **채택하지 않을 부분**:
  - Post-hoc 분석 — 본 프로젝트는 학습 중 dynamic 우선. mask 학습을 학습 중에 통합해야 함
- **후속 실험 가설**:
  - Phase 1 의 dead block 에 mask 적용 시 mask 가 거의 모두 0 으로 학습 → dead 가 functionally 확정
  - 사용자 컨셉의 function-level node 중 attention head 가 가장 effective 한가 (Csordás 의 발견 따라)

## 참고 / 인용

- 공식 코드: <https://github.com/RobertCsordas/modules> (PyTorch)
- 관련 논문: [Modular Deep Learning](2023-modular-deep-learning-pfeiffer.md) (survey), [Circuit Compositions](https://arxiv.org/html/2410.01434v3) (후속 — Transformer circuit composition)
- 본 프로젝트 내 인용 위치: Transformer 의 emergent modularity 의 정량적 증거, function-level 노드 단위 결정 reference
