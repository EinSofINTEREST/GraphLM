---
title: "A Simple and Effective Pruning Approach for Large Language Models"
authors: "Sun, M., Liu, Z., Bair, A., & Kolter, J. Z."
year: 2024
venue: "ICLR 2024"
url: "https://arxiv.org/abs/2306.11695"
arxiv_id: "2306.11695"
code_url: "https://github.com/locuslab/wanda"
tags: ["computation-graph", "dynamic-param", "sparse", "wanda", "post-training", "weight-activation", "llm", "simple"]
status: "draft"
cited_in: []
---

# Wanda — Weights × Activations Pruning

## TL;DR (3줄)

- LLM pruning 의 importance metric 을 **|weight| × ‖activation‖** 의 단순 곱으로 정의 — SparseGPT 의 Hessian 계산 불필요.
- Calibration 한 번의 forward pass 만 필요. SparseGPT 대비 훨씬 단순하면서 동등 또는 약간 우위.
- 본 프로젝트 관점: **\"단순한 importance signal 의 power\"** — 복잡한 Hessian 보다 weight × activation 이 충분. 본 프로젝트의 trigger / candidate scoring 단순화 reference.

> ⚠️ **주의**: SparseGPT 와 함께 **post-training pruning** — during-training 패러다임과는 시점 다름. baseline reference.

## 핵심 기여

- **Weight magnitude × activation norm**:
$$
S_{ij} = |W_{ij}| \cdot \|X_j\|_2
$$

여기서 $X_j$ 는 input feature $j$ 의 calibration data 상의 L2 norm. (원논문 §3 Eq. 1)

- **No Hessian, no second-order**: SparseGPT 의 column-wise Hessian inverse 불필요 → 구현 극단 단순.
- **No weight update after pruning**: SparseGPT 가 prune 후 나머지 weight 보정하는 반면, Wanda 는 mask 만 적용 — \"가장 단순한\" pruning.
- **Per-output comparison**: importance 비교를 row-wise (같은 output neuron 의 input weight 끼리) — 더 fair.

## 방법 요약

- 데이터: C4 의 작은 calibration set (128 samples) — pruning 시.
- 모델: LLaMA, LLaMA-2 family (7B / 13B / 30B / 65B).
- 학습:
  1. Calibration: 128 samples forward → activation norm 추출
  2. 각 layer 의 weight $W_{ij}$ 마다 $S_{ij}$ 계산
  3. Row-wise top-$k$ 선택 (50% sparse)
  4. Mask 적용. 보정 없음.

## 실험 / 결과

- LLaMA-65B 50% sparsity: perplexity 4.57 vs dense 4.45 (+0.12) — SparseGPT 와 거의 동등.
- 7B / 13B / 30B 모두 SparseGPT 동등 또는 약간 우위.
- **구현 코드 ~30줄** — SparseGPT 의 수백 줄 대비 극단 단순.
- 동일 calibration set 에서 SparseGPT 보다 약 10x 빠름 (Hessian 계산 없음).
- 재현성: 공식 PyTorch (locuslab) 공개.

## 한계 / 비판적 시각

- **Post-training** — during-training 패러다임 외.
- Activation calibration set 분포 의존 — out-of-distribution 데이터에 약화 가능.
- 50% sparsity 위주 검증 — 70% 이상에서 SparseGPT 와의 비교 부족.
- Structured sparsity (2:4) 에서는 SparseGPT 가 약간 우위.

## 본 프로젝트 시사점

- **\"단순한 importance metric 의 충분성\"** 이 본 프로젝트에 가장 큰 시사 — Firefly 의 functional gradient 같은 복잡한 신호 대신 weight × activation 의 단순 곱이 baseline 으로 충분할 가능성.
- **차용할 아이디어**:
  - **Weight × activation importance** — 본 프로젝트의 candidate scoring (어느 head/FFN 을 제거/유지) 에 직접 활용. RigL 의 magnitude-only 보다 정교, Firefly 의 gradient 보다 단순.
  - **Calibration 한 번 forward** — 비용 매우 작음. 본 프로젝트의 \"학습 중 주기적 평가\" 에 negligible overhead.
  - **Row-wise comparison** — output neuron 별 fair pruning. 본 프로젝트의 head/expert 단위 비교에 reference.
- **채택하지 않을 부분**: post-training 전체 paradigm — during-training 으로 재해석 필요.
- **후속 실험 가설**:
  - 본 프로젝트의 \"학습 중 trigger\" 에 Wanda 의 W×A criterion 을 사용 — Firefly 의 gradient 대비 quality / 비용 비교.
  - Wanda + RigL hybrid — magnitude prune (Wanda) + gradient regrow (RigL) 의 결합 가능성.

## 참고 / 인용

- 공식 코드: <https://github.com/locuslab/wanda> (PyTorch)
- 관련 논문: [SparseGPT](2023-sparsegpt-frantar.md) (직전 generation, Hessian 기반), [RigL](2020-rigl-evci.md) (during-training 비교)
- 본 프로젝트 내 인용 위치: 추후 simple importance scoring 실험에서
