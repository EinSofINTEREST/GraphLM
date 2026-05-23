---
title: "AutoGrow: Automatic Layer Growing in Deep Convolutional Networks"
authors: "Wen, W., Yan, F., Chen, Y., & Li, H."
year: 2020
venue: "AAAI 2020"
url: "https://arxiv.org/abs/1906.02909"
arxiv_id: "1906.02909"
code_url: ""
tags: ["computation-graph", "dynamic-param", "adaptive-trigger", "autogrow", "depth-growth", "training-signal", "convergence-trigger"]
status: "draft"
cited_in: []
---

# AutoGrow — Automatic Depth Growth via Training Signal

## TL;DR (3줄)

- CNN 의 깊이를 **수동 설계 없이** 학습 신호로 자동 결정 — 학습 중 validation accuracy 가 수렴 (\"meta-cycle\" 종료) 하면 layer 추가, 더 이상 개선 없으면 중단.
- 4가지 growth policy (\"sub-network growing\", \"network morphism\" 등) 비교 — function-preserving 변형이 가장 안정.
- 본 프로젝트 관점의 핵심: **\"성장의 trigger 가 학습 곡선의 plateau 검출\"** 의 명시적 reference. CNN 위주지만 패턴은 Transformer 로 일반화 가능.

## 핵심 기여

- **Adaptive depth decision**: 사람이 layer 수를 정하지 않고, validation accuracy 의 수렴 신호로 layer 추가 시점 결정.
- **Meta-cycle framework**: 한 cycle = (학습 → 수렴 검출 → 성장 또는 종료). 종료는 \"성장 후에도 accuracy 정체\" 신호.
- **4 growth operators 비교**:
  1. Random init (function 안 보존)
  2. Zero init (residual scale 0)
  3. Net2DeeperNet (identity)
  4. Sub-network growing (작은 sub-block 부터)
- Function-preserving (zero init / Net2DeeperNet) 이 압도적 안정. Random 은 학습 spike.

## 방법 요약

- 데이터: CIFAR-10/100, ImageNet (ResNet 류).
- 모델: ResNet / VGG / WideResNet 변형. 초기 작은 모델 → AutoGrow 가 깊이 자동 결정.
- 학습:
  1. 작은 backbone 으로 시작 (예: 4-layer)
  2. Validation accuracy 의 sliding window 평균이 plateau (변화 $< \epsilon$) → trigger
  3. 새 layer 추가 (4 policy 중 하나)
  4. 새 layer 학습 → 다음 plateau 검출 → 반복
  5. 성장 후에도 plateau 가 즉시 다시 발생 → 종료
- 핵심 흐름 (trigger):

window $W$ epoch 의 validation accuracy 표준편차:
$$
\sigma_W < \epsilon \;\Rightarrow\; \text{trigger\_grow} = \text{True}
$$

성장 후 $\sigma_W < \epsilon$ 이 즉시 재발생하면 → 종료.

## 실험 / 결과

- CIFAR-10 ResNet: AutoGrow 가 발견한 깊이가 사람이 정한 ResNet-110 동등 또는 우위 + 학습 시간 단축.
- CIFAR-100 / ImageNet 유사 패턴.
- 4 policy 비교: function-preserving (zero / Net2DeeperNet) 이 random / sub-network 대비 안정성 + 최종 quality 우위.
- 재현성: 공식 코드 미공개 (논문 detail 만), 후속 재구현 다수.

## 한계 / 비판적 시각

- **CNN-only 검증** — Transformer / LM 적용 미보고 (본 프로젝트의 직접 적용은 별도 검증 필요).
- Plateau threshold $\epsilon$ + window $W$ 가 hyperparameter — 학습 진동 (noise) 에 false positive 가능.
- Depth-only growth — width / FFN dim 등 다른 axis 미포함 (CompoundGrowth 가 보완).
- 종료 criterion 이 \"성장 후 즉시 plateau\" — 더 정교한 신호 (예: validation loss trajectory) 활용 여지.

## 본 프로젝트 시사점

- **본 프로젝트 trigger 의 직접 reference** — \"plateau 검출 → grow\" 패턴이 가장 직관적 + 구현 쉬움.
- **차용할 아이디어**:
  - **Sliding-window variance 기반 plateau 검출** — validation accuracy 또는 loss 의 $\sigma_W < \epsilon$ 단순 trigger. 본 프로젝트 첫 trigger 구현의 default.
  - **Function-preserving growth 의 우월성 검증** — random init 의 spike 방지가 명시적 — 본 프로젝트도 LiGO / Net2Net 의 function preservation 사용 정당화.
  - **Auto-termination** — 성장 후 또 plateau 면 자동 멈춤. infinite growth 방지.
- **채택하지 않을 부분**: CNN-specific layer types — Transformer block 으로 직접 대체.
- **후속 실험 가설**:
  - AutoGrow 의 plateau trigger 를 Transformer 의 사전학습에 적용 — validation perplexity 의 plateau 가 \"성장 시점\" 으로 reliable 한지.
  - $W$ (window) 와 $\epsilon$ (threshold) 의 sweep — small dataset 에서의 sensitivity 측정.
  - Width + depth + FFN 의 multi-axis 확장 시 trigger 가 어느 axis 를 성장할지 결정하는 메커니즘 (DEN 의 group-sparsity 와 합성 가능).

## 참고 / 인용

- 공식 코드: 없음 (논문 detail 기반 후속 재구현)
- 관련 논문: [DEN](2018-den-yoon.md) (다른 trigger 종류), [Firefly NAD](2020-firefly-wu.md) (gradient 기반 trigger), [Self-Expanding NN](2024-self-expanding-mitchell.md) (modern 일반화), [Net2Net](2016-net2net-chen.md) (function-preserving 의 origin)
- 본 프로젝트 내 인용 위치: 추후 plateau-trigger growing Transformer 실험 노트북에서
