---
title: "AutoFormer: Searching Transformers for Visual Recognition"
authors: "Chen, M., Peng, H., Fu, J., & Ling, H."
year: 2021
venue: "ICCV 2021"
url: "https://arxiv.org/abs/2107.00651"
arxiv_id: "2107.00651"
code_url: "https://github.com/microsoft/Cream/tree/main/AutoFormer"
tags: ["computation-graph", "autoformer", "nas", "transformer", "weight-entanglement", "supernet"]
status: "draft"
cited_in: []
---

# AutoFormer — Neural Architecture Search for Transformers

## TL;DR (3줄)

- ViT 의 hyperparameter (embedding dim, head 수, MLP ratio, depth) 를 **search space** 로 정의하고, 단일 supernet 학습으로 다양한 subnet 을 동시에 학습하는 NAS framework.
- **Weight entanglement** 기법으로 같은 layer 의 서로 다른 hyperparameter 조합이 weight 를 공유 → supernet 학습이 안정.
- 본 프로젝트 패러다임에서 **\"Transformer 아키텍처 자체가 search graph\"** — 노드는 sub-module 후보, edge 는 module 간 연결 선택.

## 핵심 기여

- **Transformer-specific search space**: ViT 의 각 layer 마다 (embed_dim, head 수, MLP ratio) 의 후보 집합 + depth 의 후보. 약 $10^{17}$ 개 architecture.
- **Weight entanglement**: 동일 layer 의 더 큰 sub-module 이 작은 sub-module 의 weight 를 \"포함\" (slice) → supernet 의 모든 subnet 이 동시 학습 가능, 후속 fine-tune 불필요.
- **One-shot NAS for Vision Transformer**: 단일 supernet 학습 후 evolutionary search 로 best subnet 추출 — search cost $O(1)$ in subnets.

## 방법 요약

- 데이터: ImageNet-1k.
- 모델: ViT 변형의 search space (3 size 군 — tiny/small/base, 각 군 안에 $10^{17}$ 후보).
- 학습:
  1. Supernet 학습 (랜덤 subnet sampling per batch, weight entanglement 로 공유)
  2. Evolutionary search (mutation + crossover) 로 best subnet 발견
  3. 발견된 subnet 으로 standalone inference (별도 fine-tune 불필요)
- 핵심 아이디어 (weight entanglement):

큰 hidden dim 의 weight $W_{\text{large}}$ 가 있을 때, 작은 hidden dim 의 weight $W_{\text{small}} = W_{\text{large}}[:d_{\text{small}}, :d_{\text{small}}]$ — slice 로 공유.

## 실험 / 결과

- AutoFormer-base ImageNet top-1: **82.4%** (ViT-Base 77.9%, DeiT-Base 81.8% 대비 우위).
- AutoFormer-tiny 74.7% (DeiT-Tiny 72.2%).
- Search cost: NASNet 류 (수천 GPU-day) 대비 1/100~1/1000.
- 재현성: Microsoft Cream repo (PyTorch) 공개.

## 한계 / 비판적 시각

- 검증이 **vision (ImageNet)** 위주 — NLP / LM 에 직접 적용은 미보고 (후속 작업 필요).
- Weight entanglement 가 search space 의 \"중첩 가능성\" 을 전제 — 임의 구조 변화 (skip connection 추가 등) 에는 적용 어려움.
- Supernet 학습 자체가 dense 모델 한 번 학습 비용과 비슷 → 정말로 cost free 는 아님.
- 발견된 best subnet 이 \"좋다\" 의 metric 이 ImageNet 정확도 1개 — multi-objective (latency, memory) 는 별도.

## 본 프로젝트 시사점

- **Architecture-as-graph 접근의 vision 측 대표** — 본 프로젝트가 Transformer 의 sub-component 선택 (head 수, expert 수, layer 수 등) 을 학습 대상으로 본다면 본 논문의 weight entanglement 가 핵심 트릭.
- **차용할 아이디어**:
  - **Weight slicing for shared search space** — 작은 모델 weight 를 큰 모델의 sub-slice 로 정의해 동시 학습. 본 프로젝트의 MoE expert 수 변동 실험에 응용 가능.
  - **Evolutionary search** — gradient 가 흐르지 않는 discrete 결정 (expert 활성 여부 등) 의 한 가지 학습 방법.
- **채택하지 않을 부분**: vision-specific search space (patch size 등) — 본 프로젝트는 NLP / LM 위주.
- **후속 실험 가설**: Switch Transformer 의 expert 수 ($N \in \{4, 8, 16, 32\}$) 와 top-$k$ ($k \in \{1, 2\}$) 의 조합을 supernet-style 로 동시 학습한 뒤 best configuration 을 evolutionary search 로 추출하는 micro-experiment.

## 참고 / 인용

- 공식 코드: <https://github.com/microsoft/Cream/tree/main/AutoFormer>
- 관련 논문: [GHN-3](2023-ghn3-knyazev.md) (NN 을 graph 로 본 또 다른 접근), Once-for-All (Cai et al., ICLR 2020) — 다른 NAS 패러다임
- 본 프로젝트 내 인용 위치: 추후 architecture search 실험 노트북에서
