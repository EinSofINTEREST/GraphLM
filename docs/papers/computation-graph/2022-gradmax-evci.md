---
title: "GradMax: Growing Neural Networks using Gradient Information"
authors: "Evci, U., van Merriënboer, B., Unterthiner, T., Vladymyrov, M., & Pedregosa, F."
year: 2022
venue: "ICLR 2022"
url: "https://arxiv.org/abs/2201.05125"
arxiv_id: "2201.05125"
code_url: "https://github.com/google-research/growneuron"
tags: ["computation-graph", "dynamic-param", "adaptive-trigger", "gradmax", "gradient-based", "growth", "modern", "function-preserving"]
status: "draft"
cited_in: []
---

# GradMax — Gradient-Maximizing Neural Growth

## TL;DR (3줄)

- 새 neuron 의 init weight 를 **\"gradient norm 을 최대화\"** 하는 방향으로 결정 — 추가된 neuron 이 학습 초기부터 가장 빠르게 loss 감소에 기여.
- Net2Net 의 function preservation (output 변경 없음) + Firefly 의 gradient signal 둘 다 만족 — 두 갈래의 결합.
- 본 프로젝트 관점: **\"성장 직후 학습 효율을 극대화\"** 의 modern recipe. RigL 저자의 후속작 — DST 와 growth 의 통합 시각.

## 핵심 기여

- **Gradient-maximizing initialization**: 새 neuron 의 weight 를 \"output 영향 0 (function preservation)\" 제약 하에서 \"|gradient| 최대화\" 로 init. 단순 zero / random init 대비 학습 가속.
- **Function-preserving 분석**: 새 column / row 를 zero 로 init 하면 forward 는 동일, gradient 는 가장 큰 방향으로 결정 → 첫 step 부터 의미 있는 update.
- **Closed-form solution**: gradient 의 SVD top-component 로 새 weight 의 최적 init 계산 — 학습 가능 operator 없이 analytical.
- **Multiple growth axes**: width (neuron 추가) + depth (layer 추가) 모두 지원.

## 방법 요약

- 데이터: CIFAR-10/100, ImageNet (ResNet 류).
- 모델: MLP / VGG / ResNet 변형 — 작은 backbone 으로 시작.
- 학습:
  1. 작은 model 학습 (수렴)
  2. Growth step:
     - Function-preserving constraint 하에서 새 column $u$ 의 init: $u^* = \arg\max_u \|\nabla \mathcal{L} \cdot u\|^2$
     - SVD 의 top singular vector 로 closed-form 계산
  3. 새 weight 와 함께 학습 계속
- 핵심 흐름 (원논문 §3, gradient-maximizing init):

새 weight $u$ 의 optimal init:
$$
u^* = \arg\max_u \|\nabla_{u} \mathcal{L}\|^2 \quad \text{s.t.}\quad f_{\text{after}}(x) = f_{\text{before}}(x) \;\forall x
$$

Constraint 하 SVD top component 선택.

## 실험 / 결과

- CIFAR-10/100 ResNet: GradMax 가 random / Net2Net init 대비 성장 직후 수렴 속도 일관 우위 (+30~50% step 절감).
- ImageNet 유사 패턴.
- Firefly / Net2Net 의 hybrid 로 볼 수 있음 — 두 baseline 모두 동등 이상.
- 재현성: 공식 PyTorch (Google research) 공개.

## 한계 / 비판적 시각

- Gradient 의 SVD 계산 — 큰 layer 에서 비용. low-rank approximation 으로 완화 가능하나 quality 영향.
- **CNN 위주 검증** — Transformer 의 attention / FFN 에서 \"function-preserving + gradient 최대화\" 의 closed-form 이 어떻게 변하는지 미보고.
- Growth schedule (언제 키울지) 자체는 GradMax 의 범위 밖 — 결합할 trigger 필요 (AutoGrow / Firefly 등).
- Layer-level depth growth 보다 width growth 에서 효과 두드러짐.

## 본 프로젝트 시사점

- **\"성장 후 즉시 학습 가속\" 의 modern reference** — 본 프로젝트가 Stacking / LiGO / Net2Net 의 단순 init 대비 quality 우위 추구 시 GradMax 가 최선의 시작 init.
- **차용할 아이디어**:
  - **Function-preserving + gradient-maximizing init** — 두 제약의 결합. 본 프로젝트의 growth operator 가 quality 와 안정성 동시 확보.
  - **Closed-form solution** (SVD) — 학습 가능 operator (LiGO) 보다 단순. 작은 toy 실험에서 우선.
  - **DST + growth 통합 시각** — RigL 저자가 같이 다루는 의도 — 본 프로젝트도 sparse + growth hybrid 의 통일된 프레임워크 가능성.
- **채택하지 않을 부분**: CNN-specific layer types — Transformer 의 multi-head attention 으로 옮길 때 \"gradient 최대화 + head 의 function preservation\" 의 적절한 정의 필요.
- **후속 실험 가설**:
  - Transformer FFN dim 확장 시 GradMax 의 init 이 random / zero / LiGO 학습 가능 대비 quality / 수렴 속도 비교.
  - Firefly (어디 키울지) + GradMax (어떻게 init) + AutoGrow (언제 키울지) 의 3-way 결합 — 가장 완전한 trigger + operator 조합.

## 참고 / 인용

- 공식 코드: <https://github.com/google-research/growneuron> (PyTorch, Google research)
- 관련 논문: [Firefly NAD](2020-firefly-wu.md) (gradient-based 다른 형태), [Net2Net](2016-net2net-chen.md) (function preservation origin), [RigL](2020-rigl-evci.md) (저자 같음, DST 시각)
- 본 프로젝트 내 인용 위치: 추후 gradient-maximizing init 실험 노트북에서
