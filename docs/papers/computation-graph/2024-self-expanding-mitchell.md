---
title: "Self-Expanding Neural Networks"
authors: "Mitchell, R., Mündler, N., Menzel, A., Caterini, A., & Mitchell, T."
year: 2024
venue: "arXiv preprint"
url: "https://arxiv.org/abs/2307.04526"
arxiv_id: "2307.04526"
code_url: "https://github.com/Rmitch9000/Self-Expanding-Neural-Networks"
tags: ["computation-graph", "dynamic-param", "adaptive-trigger", "self-expanding", "rate-of-change", "natural-gradient", "modern"]
status: "draft"
cited_in: []
---

# Self-Expanding Neural Networks — Rate-Based Adaptive Growth

## TL;DR (3줄)

- 신경망 성장의 trigger 를 **\"loss 의 natural rate-of-change\"** 라는 단일 metric 으로 정식화 — Fisher information / natural gradient 기반.
- Hyperparameter (loss threshold, plateau window) 없이 \"성장하면 더 큰 loss 감소가 기대되는가\" 를 직접 측정 → 자동 trigger.
- 본 프로젝트의 trigger 비전 (\"학습 신호 기반 적응적 확장\") 의 가장 modern formulation. 2024 preprint 라 후속 검증은 진행 중.

## 핵심 기여

- **Natural rate-of-change criterion**: 현 capacity 로 loss 가 더 이상 빠르게 감소하지 않으면 (rate of natural-gradient descent 가 감소) → 성장 trigger. Fisher information 으로 정식화.
- **Hyperparameter-free trigger** (이론적): 기존 DEN / AutoGrow 의 threshold $\tau$, window $W$ 등의 hyperparameter 없이 작동.
- **Width + depth 동시 의사결정**: 어느 axis 를 키울지도 natural gradient 기반.
- **Convergence guarantee** (proposition): 정해진 max capacity 에 도달하기 전 stop 보장 — infinite growth 방지.

## 방법 요약

- 데이터: 작은 합성 dataset, MNIST, 단순 regression task (저자 표현으로는 \"proof-of-concept\" 단계).
- 모델: 작은 MLP — proof of concept 위주, 큰 모델 적용은 추후.
- 학습:
  1. 작은 init 으로 시작
  2. Standard SGD step
  3. Periodic check: rate of natural gradient $\partial \mathcal{L} / \partial F$ ($F$ = Fisher information) 계산
  4. Rate 가 threshold 이하 → growth (axis 선택도 동일 metric 으로)
  5. 새 capacity 와 함께 학습 계속
- 핵심 흐름:

자연 gradient 의 rate-of-change criterion:
$$
\text{trigger\_grow} = \frac{\partial \mathcal{L}}{\partial F} < \tau_{\text{natural}}
$$

(저자의 명시: $\tau_{\text{natural}}$ 은 unit-less, dataset-independent 한다고 주장.)

## 실험 / 결과

- 작은 dataset / 모델 — proof-of-concept 수준의 검증.
- Hyperparameter-free trigger 가 의미 있는 시점에서 성장 — AutoGrow / DEN baseline 대비 anecdotal 우위.
- 큰 dataset / 모델 적용은 미보고 — 본 논문이 framework 정립 단계.
- 재현성: 공식 PyTorch 공개. 후속 follow-up 작업 진행 중.

## 한계 / 비판적 시각

- **검증 규모가 작음** — proof-of-concept 단계. Transformer / LM 의 큰 scale 적용 미확인.
- Natural gradient 의 Fisher information 계산 비용 — full Fisher 는 $O(N^2)$, approximation 이 필요.
- Hyperparameter-free 주장이 이상적 — 실제로는 Fisher approximation 의 hyperparameter (e.g., damping) 가 숨어 있음.
- 2024 preprint — 학계 검증 진행 중, replication 사례 적음.

## 본 프로젝트 시사점

- **가장 modern trigger formulation** — 본 프로젝트가 \"unified trigger criterion\" 을 추구할 때 reference. DEN/AutoGrow 가 단편적 신호 (loss / plateau) 라면 Self-Expanding 은 \"성장의 marginal utility\" 를 직접 측정.
- **차용할 아이디어**:
  - **Marginal utility 의 명시적 측정** — \"성장하면 얼마나 좋아지는가\" 를 trigger 의 기본 quantity 로. 본 프로젝트 trigger 설계의 conceptual reference.
  - **Multi-axis decision** — 어느 axis (depth / width / FFN dim) 를 키울지를 동일 metric 으로 통합 결정.
  - **Convergence guarantee** — infinite growth 방지의 형식적 분석. 본 프로젝트의 안전성 contract.
- **채택하지 않을 부분**: Fisher information 의 full 계산 — 큰 모델에 비현실. K-FAC / EKFAC 등 approximation 이 필요.
- **후속 실험 가설**:
  - Self-Expanding 의 natural rate criterion 을 Transformer 의 작은 setup 에서 검증 — Fisher approximation 의 quality 가 trigger 의 reliability 에 미치는 영향.
  - Hyperparameter-free 주장의 실증 — 실제로 dataset 간 transfer 시 추가 튜닝 없이 작동하는지.
  - DEN / AutoGrow / Firefly / Self-Expanding 의 4 trigger 비교 ablation — 어느 신호가 small Transformer 사전학습에서 가장 reliable 한지.

## 참고 / 인용

- 공식 코드: <https://github.com/Rmitch9000/Self-Expanding-Neural-Networks> (PyTorch)
- 관련 논문: [DEN](2018-den-yoon.md) (loss-based 단편 trigger), [AutoGrow](2020-autogrow-wen.md) (plateau-based 단편), [Firefly NAD](2020-firefly-wu.md) (gradient-based where-to-grow)
- 본 프로젝트 내 인용 위치: 추후 unified trigger formulation 실험 노트북에서
