---
title: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
authors: "Frantar, E., & Alistarh, D."
year: 2023
venue: "ICML 2023"
url: "https://arxiv.org/abs/2301.00774"
arxiv_id: "2301.00774"
code_url: "https://github.com/IST-DASLab/sparsegpt"
tags: ["computation-graph", "dynamic-param", "sparse", "sparsegpt", "post-training", "one-shot", "llm", "modern"]
status: "draft"
cited_in: []
---

# SparseGPT — One-Shot Post-Training LLM Pruning

## TL;DR (3줄)

- 사전학습된 거대 LLM (OPT-175B, BLOOM-176B) 을 **단일 GPU 한 번의 forward + 4시간 이내** 로 50% sparsity 까지 prune — 추가 학습 없이.
- Optimal Brain Surgeon (OBS) 의 LLM-scale 적응 — Hessian inverse 의 column-wise approximation.
- 본 프로젝트 관점: **post-training direction (during-training 과 다름)** 이지만 Transformer sparsity 분야의 mile stone — \"훈련 끝난 모델을 어떻게 줄일까\" 의 reference. 본 프로젝트의 training-time 패러다임과 dual.

> ⚠️ **주의**: SparseGPT 는 **post-training pruning** 으로 본 프로젝트의 \"학습 중 동적 파라미터 수\" 패러다임과 시점이 다름. baseline / 비교 reference 로만 활용 — 직접 채택 X.

## 핵심 기여

- **One-shot pruning at LLM scale**: 175B 모델을 단일 GPU 에서 4시간 안에 50% sparse 화. 이전 방법은 LLM 에 비현실적.
- **Column-wise Hessian approximation**: 전체 Hessian 은 LLM 에 너무 큼. 한 column 씩 분리해 OBS-style 보정.
- **Layer-wise 독립 처리**: 각 Transformer layer 가 sequential 하게 pruned + corrected — 메모리 효율적.
- **Unstructured + Semi-structured (2:4) sparsity** 모두 지원. 2:4 는 GPU sparse tensor core 가속 가능.

## 방법 요약

- 데이터: pruning 시 calibration set (C4 의 작은 sample). 추가 학습 없음.
- 모델: OPT-125M ~ 175B, BLOOM-176B.
- 학습:
  1. Pretrained LLM 의 각 layer 의 weight matrix $W$ 순회
  2. 각 column 마다 OBS-style optimal pruning + 나머지 column 의 보정 update
  3. 50% sparse mask 적용
- 핵심 흐름 (원논문 §3):

각 weight $W_{ij}$ 의 \"제거 시 reconstruction error\" 추정:
$$
\epsilon_{ij} = \frac{W_{ij}^2}{[H^{-1}]_{ii}}
$$

가장 작은 $\epsilon_{ij}$ 부터 제거 + 나머지 weight 의 closed-form 보정.

## 실험 / 결과

- OPT-175B 50% sparsity: perplexity 8.43 (dense 8.34, +0.09).
- BLOOM-176B 50%: perplexity 13.34 (dense 13.34, 거의 동등).
- 4 시간 / 단일 A100 GPU — 이전 SOTA (수일~수주) 대비 극적 단축.
- 60% sparsity 까지 quality 거의 유지, 70% 부터 급격 저하.
- 재현성: 공식 PyTorch (IST-DASLab) 공개.

## 한계 / 비판적 시각

- **Post-training only** — 학습 끝난 모델에 적용. 학습 중 dynamic 과는 무관.
- Calibration set 의존 — set 의 분포가 다운스트림과 다르면 quality 영향.
- 70% 이상 extreme sparsity 에서 quality drop (Wanda / DST 류가 개선).
- Hessian approximation 의 column-wise 가정 → 정확하지 않은 layer 에서 quality 손실 가능.

## 본 프로젝트 시사점

- **본 프로젝트와 직접적 alignment 약함** — post-training paradigm. 단, Transformer sparsity 분야의 modern landmark 라 reference 가치 큼.
- **차용할 아이디어 (제한적)**:
  - **Column-wise OBS approximation** — 본 프로젝트의 \"학습 중 prune\" 신호로 OBS-style importance 를 활용 가능. 단순 |weight| 보다 정교한 importance signal.
  - **Layer-wise 독립 처리** — 메모리 효율 — 본 프로젝트의 큰 모델 다룰 때 reference.
- **채택하지 않을 부분**: post-training 전체 paradigm — 본 프로젝트는 during-training.
- **후속 실험 가설**:
  - SparseGPT 의 column-wise OBS criterion 을 학습 중 (= during-training) RigL-style cycle 에 통합 가능한지 — 더 정확한 prune signal.
  - 본 프로젝트의 grown 모델 사후에 SparseGPT 적용 — 학습 후 추가 1-step 압축의 dual operator.

## 참고 / 인용

- 공식 코드: <https://github.com/IST-DASLab/sparsegpt> (PyTorch)
- 관련 논문: [Wanda](2024-wanda-sun.md) (다음 generation, 더 단순), [Top-KAST](2020-top-kast-jayakumar.md) (during-training 대비)
- 본 프로젝트 내 인용 위치: 추후 post-training 비교 baseline 으로
