---
title: "Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks"
authors: "Wu, L., Wang, D., & Liu, Q."
year: 2020
venue: "NeurIPS 2020"
url: "https://arxiv.org/abs/2102.08574"
arxiv_id: "2102.08574"
code_url: "https://github.com/klightz/Firefly"
tags: ["computation-graph", "dynamic-param", "adaptive-trigger", "firefly", "gradient-based", "node-selection", "splitting"]
status: "draft"
cited_in: []
---

# Firefly NAD — Gradient-Based Growth Decision

## TL;DR (3줄)

- 신경망 성장을 \"functional space 의 gradient descent\" 로 정식화 — **각 후보 새 neuron 의 expected loss 감소량을 gradient 로 추정** 해 가장 유망한 후보만 추가.
- Width growth (neuron 추가) + neuron splitting (한 neuron 을 둘로 나누어 표현력 확장) 의 두 operator 모두 지원.
- 본 프로젝트의 \"trigger 가 학습 신호 (gradient) 기반\" 패러다임의 가장 직접적 reference. \"어디를 키울지\" 를 학습된 함수로 결정.

## 핵심 기여

- **Functional gradient view of growth**: 한 neuron 추가의 expected loss 변화량을 first-order Taylor approximation:

$$
\Delta \mathcal{L} \approx -\eta \cdot \|\nabla_\theta \mathcal{L} \cdot \phi_{\text{new}}\|^2
$$

여기서 $\phi_{\text{new}}$ 는 새 neuron 의 functional contribution. gradient 가 큰 후보일수록 loss 감소 기대 큼.

- **Candidate node 평가 + top-$k$ 추가**: 매 growth step 에서 다수 후보 (각 candidate 의 functional contribution 다름) 의 \"expected loss 감소\" 계산 → 상위 $k$ 만 실제 추가.
- **Neuron splitting**: 단순 추가 외에, 기존 neuron 을 두 개로 분할 (서로 다른 init perturbation) → \"같은 기능을 다양화\" 확장.
- **Generic framework**: MLP, CNN, RNN 에 모두 적용. operator 가 architecture-agnostic.

## 방법 요약

- 데이터: MNIST, CIFAR-10/100, PTB language modeling.
- 모델: 다양 (MLP / VGG / ResNet / LSTM).
- 학습:
  1. 작은 backbone 으로 시작 (예: 2-layer)
  2. 학습 진행 중 정기적 growth step:
     - 다수의 candidate new node 의 functional gradient 계산
     - Top-$k$ 후보를 추가 / split
  3. 추가된 노드와 함께 학습 계속
  4. 종료: 최대 모델 크기 도달 또는 추가 효과 미미 시
- 핵심 흐름 (candidate selection):

$$
i^* = \arg\max_i \left| \langle \nabla_\theta \mathcal{L},\, \phi_i \rangle \right|
$$

각 후보 $i$ 의 functional contribution $\phi_i$ 와 loss gradient 의 내적이 최대인 것 선택.

## 실험 / 결과

- CIFAR-10 ResNet 류: Firefly 가 발견한 architecture 가 동일 FLOP budget 의 hand-designed network 동등 또는 우위.
- Neuron splitting 의 효과: 단순 random 추가 대비 +1~2% accuracy.
- PTB LM (LSTM): perplexity 우위.
- 재현성: 공식 PyTorch 공개.

## 한계 / 비판적 시각

- **Gradient 계산 비용** — 모든 candidate 의 functional gradient 가 필요 → forward/backward 추가 비용. \"sparse training\" 의 효율 일부 상쇄.
- Candidate set 의 정의가 architecture-dependent — Transformer 의 \"FFN expert 추가\" 등 본 프로젝트 setting 으로 일반화는 추가 작업 필요.
- 평가가 vision / small LM — 큰 Transformer 미적용.
- Splitting 의 \"diversification 효과\" 가 직관적이나 이론적 근거 약함 — heuristic 측면.

## 본 프로젝트 시사점

- **\"학습 신호 기반 어디를 키울지 결정\" 의 가장 직접적 reference** — 본 프로젝트의 trigger 시스템이 \"plateau → grow\" (when) 만이 아니라 \"어느 FFN/head 를 키울지\" (where) 까지 학습 신호로 결정한다면 Firefly 의 functional gradient 가 reference.
- **차용할 아이디어**:
  - **Functional gradient 로 candidate scoring** — 본 프로젝트의 \"어느 layer 를 키울지\" 결정에 직접 활용 가능.
  - **Top-$k$ candidate 만 추가** — 한 번에 모두 추가가 아니라 가장 유망한 일부만 → 과확장 방지.
  - **Splitting operator** — 단순 추가 외에 기존 neuron 을 두 변형으로 분리하는 design space.
- **채택하지 않을 부분**: 모든 candidate 의 매 step gradient 계산 — 비용 큼. 본 프로젝트는 작은 후보 집합 / 주기적 평가로 단순화.
- **후속 실험 가설**:
  - Firefly 의 functional gradient criterion 이 Transformer 의 \"새 FFN expert 추가\" 결정에 의미 있는 신호인지 — MoE expert 의 utilization 과 상관관계.
  - DEN 의 loss-threshold + AutoGrow 의 plateau detection + Firefly 의 functional gradient — 세 trigger 의 stacking / ensemble.

## 참고 / 인용

- 공식 코드: <https://github.com/klightz/Firefly> (PyTorch)
- 관련 논문: [DEN](2018-den-yoon.md) (loss-based trigger), [AutoGrow](2020-autogrow-wen.md) (plateau-based trigger), [Self-Expanding NN](2024-self-expanding-mitchell.md) (modern rate-based)
- 본 프로젝트 내 인용 위치: 추후 gradient-based growth 실험 노트북에서
