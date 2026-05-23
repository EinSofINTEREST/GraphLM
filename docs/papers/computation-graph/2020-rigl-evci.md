---
title: "Rigging the Lottery: Making All Tickets Winners"
authors: "Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E."
year: 2020
venue: "ICML 2020"
url: "https://arxiv.org/abs/1911.11134"
arxiv_id: "1911.11134"
code_url: "https://github.com/google-research/rigl"
tags: ["computation-graph", "dynamic-param", "rigl", "dst", "sparse", "gradient-regrow", "modern-dst"]
status: "draft"
cited_in: []
---

# RigL — Gradient-Based Dynamic Sparse Training

## TL;DR (3줄)

- SET 의 random regrowth 를 **gradient magnitude 기반 regrowth** 로 개선 — \"이번 epoch 에서 학습 신호가 강한 connection\" 을 우선 활성화.
- 동일 sparsity 에서 dense 학습 후 prune (Lottery Ticket Hypothesis) 류 대비 큰 폭 quality 우위. 80~99% sparsity 에서 dense 동등 도달.
- 본 프로젝트의 **DST 갈래 modern recipe** — connection 수준의 dynamic param count 에서 quality / 효율 balance 의 reference.

## 핵심 기여

- **Magnitude prune + gradient regrow** 의 cycle:
  - **Prune** (drop step): magnitude 가 작은 connection 의 일정 fraction 제거.
  - **Regrow** (grow step): 비활성 connection 중 |gradient| 가 큰 것을 동일 수만큼 활성화.
- **Cosine schedule for prune fraction**: 학습 초반엔 큰 변동, 후반엔 mask 안정화.
- **ERK initialization** (Erdős–Rényi-Kernel): SET 의 ER 을 layer-wise sparsity 에 적응. layer 의 입력/출력 dim 에 따라 다른 sparsity 자동 할당.
- **Scaling to ResNet-50 / ImageNet** — 이전 DST 들이 작은 dataset 위주였던 한계 돌파.

## 방법 요약

- 데이터: CIFAR-10/100, ImageNet (ResNet-50), Penn Treebank (RNN).
- 모델: ResNet-50, MobileNet-V1, Conv4, RNN (LSTM).
- 학습:
  1. Sparse mask init (ERK)
  2. Standard SGD epoch
  3. $T_{\text{end}}$ 까지 매 $\Delta T$ epoch 마다:
     - drop: 작은 |weight| connection $\zeta_t$ fraction 제거
     - grow: 큰 |gradient| connection $\zeta_t$ fraction 재활성화
  4. $T_{\text{end}}$ 이후 mask 고정, 학습 계속
- 핵심 흐름 (regrow step):

$$
\text{ToGrow} = \arg\text{top-}k_{|i,j| \notin \mathcal{M}^{(t)}} \left| \frac{\partial \mathcal{L}}{\partial W_{i,j}} \right|
$$

비활성 connection 중 gradient 가 큰 top-$k$ 를 다음 epoch 에 활성화.

## 실험 / 결과

- ResNet-50 ImageNet at 80% sparsity: top-1 **75.1%** (dense baseline 76.8%, 0.5x parameter count + 0.5x FLOPs).
- 90% sparsity: 73.0% (dense 대비 -3.8%p, 0.1x param).
- SET 대비 일관 우위 — 동일 sparsity 에서 +1~3% accuracy.
- LSTM Penn Treebank: 90% sparsity 에서 dense 동등.
- 재현성: 공식 TensorFlow + Google research 공개. PyTorch 재구현 (\"rigl-torch\") 다수.

## 한계 / 비판적 시각

- **Gradient 계산 비용** — 모든 비활성 connection 의 gradient 가 필요 → dense gradient 한 번 계산해야 함. \"sparse training 효율\" 의 일부 상쇄.
- Layer-wise sparsity (ERK) 의 hyperparameter — uniform / ERK 외 다른 schedule 의 비교 부족.
- 평가가 vision 위주 — Transformer / LM 적용은 후속 (Top-KAST, Spartan 등).
- $T_{\text{end}}$ 이후 mask 고정 — \"학습 끝까지 dynamic\" 한 변형은 미보고.

## 본 프로젝트 시사점

- **DST 갈래의 modern reference** — 본 프로젝트가 connection-level dynamic param 모델 구현 시 RigL 의 prune-grow cycle 이 default recipe.
- **차용할 아이디어**:
  - **Gradient-based regrow** — 단순 random (SET) 보다 효과적. PyTorch 의 hook 으로 비활성 weight 의 grad 추적해 구현.
  - **ERK initialization** — layer 별 sparsity 자동 — 본 프로젝트의 다양한 module 크기 대응.
  - **Cosine schedule for update fraction** — 학습 후반 mask 안정화 — 본 프로젝트의 dynamic param 모델 default schedule.
- **채택하지 않을 부분**: vision-specific layer types (Conv, BN) — Transformer 의 linear / attention 으로 직접 옮길 때 ERK 의 dim 추정 공식 재유도 필요.
- **후속 실험 가설**:
  - RigL 의 magnitude prune + gradient regrow 가 Transformer 의 FFN / attention 에 적용했을 때 layer 마다 다른 sparsity sweet spot 이 나타나는지.
  - $T_{\text{end}}$ 이후에도 mask 변동을 계속 허용 (\"never freeze\") 했을 때 quality / 안정성 trade-off.

## 참고 / 인용

- 공식 코드: <https://github.com/google-research/rigl> (TensorFlow)
- 관련 논문: [SET](2018-set-mocanu.md) (DST 시초), [Net2Net](2016-net2net-chen.md) (다른 dynamic param 방향)
- 본 프로젝트 내 인용 위치: 추후 DST 실험 노트북에서
