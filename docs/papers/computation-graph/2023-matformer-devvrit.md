---
title: "MatFormer: Nested Transformer for Elastic Inference"
authors: "Devvrit, Kudugunta, S., Kusupati, A., Dettmers, T., Chen, K., Dhillon, I., Tsvetkov, Y., Hajishirzi, H., Kakade, S., Farhadi, A., & Jain, P."
year: 2023
venue: "NeurIPS 2023"
url: "https://arxiv.org/abs/2310.07707"
arxiv_id: "2310.07707"
code_url: "https://github.com/google-research/google-research/tree/master/matformer"
tags: ["computation-graph", "dynamic-param", "deployment", "matformer", "nested", "elastic", "transformer", "llm"]
status: "draft"
cited_in: []
---

# MatFormer — Nested Elastic Transformer

## TL;DR (3줄)

- 하나의 Transformer 안에 **g 개의 nested sub-transformer (g=4 typical)** 가 공존 — 가장 큰 FFN 의 첫 부분만 잘라내면 작은 FFN, 더 작은 부분이면 더 작은 FFN.
- 단일 학습으로 **모든 size 의 sub-transformer 가 동시 학습 + 사용 가능** — Once-for-All 의 LLM-친화적 modern 버전.
- 본 프로젝트 관점: **\"학습 중 dynamic param count\" 의 deployment 측 dual** — 학습된 큰 모델에서 다양 size 추출. OFA 가 vision 위주였다면 MatFormer 는 Transformer LM 첫 검증.

> ⚠️ **주의**: deployment-time flexibility 패러다임 — 학습 중 동적 증가 자체와는 dual. 본 프로젝트의 \"성장한 모델 사후 적응\" 단계에 reference.

## 핵심 기여

- **Matryoshka structure on FFN**: FFN intermediate dim 을 nested layout — 큰 FFN 의 첫 $d_1$ 채널이 가장 작은 sub-FFN, 첫 $d_2 > d_1$ 채널이 중간 sub-FFN, 전체가 가장 큰 sub-FFN.
- **All sub-transformers train together**: 매 batch 마다 random sub-transformer 선택 — 모든 sub-network 가 학습 신호 받음.
- **Mix-and-match across layers**: 추론 시 layer 마다 다른 size 의 sub-transformer 선택 가능 → exponential 한 조합 가능.
- **Same parameter, exponential subnets**: g=4, L=24 layer 면 $4^{24}$ 가능 architecture — 모두 별도 학습 없이 사용.

## 방법 요약

- 데이터: C4 (LLaMA-style pretraining), 다양 size 변형.
- 모델: MatLM (decoder LM, 78M-2.6B), MatViT (vision Transformer).
- 학습:
  1. Nested FFN init (largest size 의 random init)
  2. 매 forward step: 각 layer 마다 random sub-transformer 선택 (또는 round-robin)
  3. Standard LM loss → 모든 sub-network 동시 학습
- 핵심 흐름:

각 layer 의 g 개 sub-FFN (작은 → 큰 순서):
- $d_1 < d_2 < ... < d_g$ (예: $d_g/8, d_g/4, d_g/2, d_g$)
- $\text{FFN}_g$ 의 첫 $d_i$ 채널만 사용하면 $\text{FFN}_i$ 가 됨

## 실험 / 결과

- MatLM-2.6B (원논문 §4.1, Table 2): 학습된 supernet 에서 추출한 모든 sub-LM (78M ~ 2.6B) 이 동일 size 의 standalone 학습 동등 또는 약간 우위. 구체적 size 변형 ($d_g/8, d_g/4, d_g/2, d_g$) 은 §3.2 참조.
- Mix-and-match (layer 별 다른 size) 의 추가 자유도 — Pareto frontier 우위.
- 단일 학습 비용 = 가장 큰 sub-transformer 학습 비용 (overhead 미미).
- 재현성: 공식 (Google research) 부분 공개.

## 한계 / 비판적 시각

- **FFN 만 nested** — attention head 수 / hidden dim 은 고정. \"진정한 elastic\" 은 아님.
- 학습 시 sub-network sampling distribution 의 hyperparameter — uniform vs largest-biased 의 trade-off.
- Mix-and-match 의 exponential 조합 중 \"좋은\" 조합 검색은 추가 작업 (OFA 의 predictor 류 필요).
- 대규모 검증 (7B+) 미보고 — 후속.

## 본 프로젝트 시사점

- **\"학습 중 성장\" + \"학습 후 다양 size 추출\" 의 자연스러운 결합 후보** — 본 프로젝트가 성장된 큰 모델을 deployment 마다 다른 size 로 사용할 때 MatFormer 의 nested 구조가 직접 적용 가능.
- **차용할 아이디어**:
  - **Matryoshka FFN layout** — FFN dim 의 \"prefix subset\" 이 의미 있는 sub-FFN. 본 프로젝트의 성장된 FFN 도 nested 구조로 학습하면 deployment flexibility 부수 효과.
  - **Random sub-network sampling per batch** — 모든 size 가 학습 신호 받음. 본 프로젝트의 \"여러 candidate size 동시 학습\" 에 적용.
  - **Mix-and-match layer-wise** — 각 layer 의 다른 size 조합. 본 프로젝트의 \"layer 별 다른 capacity\" 의 deployment 측 dual.
- **채택하지 않을 부분**: head 수 고정 — 본 프로젝트는 head 도 동적이므로 추가 elastic 필요.
- **후속 실험 가설**:
  - 본 프로젝트의 grown Transformer 의 FFN 을 MatFormer 식 nested 로 재학습 시 quality 손실 / deployment flexibility 의 trade-off.
  - Adaptive growth + MatFormer nested — 성장 중에도 nested 유지하는 통합 framework 가능성.

## 참고 / 인용

- 공식 코드: <https://github.com/google-research/google-research/tree/master/matformer>
- 관련 논문: [Once-for-All](2020-once-for-all-cai.md) (vision 시대 predecessor), [LayerSkip](2024-layerskip-elhoushi.md) (다른 elastic 방향)
- 본 프로젝트 내 인용 위치: 추후 deployment flexibility 실험 노트북에서
