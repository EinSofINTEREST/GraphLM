---
title: "Information Processing in Dynamical Systems: Foundations of Harmony Theory"
authors: "Smolensky, P."
year: 1986
venue: "Parallel Distributed Processing, Vol. 1, Chapter 6 (MIT Press)"
url: "https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf"
tags: ["energy-based", "rbm-origin", "harmonium", "bipartite", "restricted-boltzmann"]
status: "draft"
cited_in: []
---

# Harmonium (RBM 원조) — bipartite restricted Boltzmann machine

## TL;DR (3줄)

- "Harmonium" 이라는 bipartite stochastic network 제안 — visible layer 와 hidden layer 의 connection 만 허용, 같은 layer 안 connection 금지 (= 후일 RBM).
- 인지과학 framework "harmony theory" 의 일환 — 표현의 *harmony* (= 음의 energy) 를 최대화.
- BM (Hinton+Sejnowski 1985) 의 fully-connected 한계를 bipartite 제약으로 완화 → conditional independence 활용한 효율적 inference 의 수학적 기반 제공.

## 핵심 기여

- **Bipartite restricted Boltzmann machine 정식화**: visible `v ∈ {0,1}^V` ↔ hidden `h ∈ {0,1}^H`, weights `W ∈ R^{V×H}`, intra-layer connection 없음.
- **Harmony function** = `-E(v, h) = Σ_ij w_ij v_i h_j + Σ_i a_i v_i + Σ_j b_j h_j`
- **Conditional independence**: bipartite 구조 덕분에 `P(h | v) = ∏_j P(h_j | v)`, `P(v | h) = ∏_i P(v_i | h)` — Gibbs sampling 한 step 이 한 layer 전체 병렬 가능.
- **인지과학적 motivation**: 인간의 perception/cognition 을 "전반적 harmony 최대화" 로 설명하는 connectionist framework.

## 방법 요약

- 데이터: cognitive task 예시 (linguistic / perceptual patterns)
- 모델: 2-layer bipartite undirected graph
  - visible `v` (data dimension)
  - hidden `h` (latent feature)
  - weight `W` (only cross-layer)
- 학습: 본 chapter 는 framework 정의가 주, 직접적 학습 알고리즘 제시 없음. BM 의 positive/negative phase contrastive 규칙 (Hinton+Sejnowski 1985) 이 자연 적용 가능하나 계산 비용 막대. 실용적 RBM 학습은 후일 Hinton 2002 의 contrastive divergence (CD-1) 로 가능해짐.
- 핵심 수식:
  $$ E(v, h) = -v^\top W h - a^\top v - b^\top h $$
  $$ P(v, h) = \frac{\exp(-E(v, h))}{Z}, \quad P(h_j = 1 | v) = \sigma(W_{:j}^\top v + b_j) $$

## 실험 / 결과

- 인지과학 framework 위주 — 양적 벤치마크보다 정성적 설명력 강조
- 실용적 학습은 후속 연구로 위임됨 (Hinton 2002 의 CD-1 이 RBM 학습의 critical breakthrough)
- 1986 당시에는 큰 impact 못 받았으나 2000년대 RBM 부활과 함께 historical foundation 으로 재조명

## 한계 / 비판적 시각

- **학습 알고리즘 부재**: 이론적 framework 만 제시, 실용적 weight update rule 없음 → 17 년간 RBM 은 거의 사용되지 않음
- **인지과학 framework 의 모호성**: harmony theory 의 더 큰 ambition (cognition 통합) 은 후속 입증 못 함
- **연결성 제약의 cost**: bipartite 제약으로 BM 대비 표현력 감소 (intra-layer correlation 표현 불가)
- 이 한계를 stack 으로 보완한 것이 후일 DBN (Hinton+Osindero+Teh 2006)

## 본 프로젝트 시사점

> Paradigm 의 **bipartite/layer-restricted graph 구조** 와 직접 정렬 — `HybridGraphLinear` 의 input ↔ output channel 도 정확히 RBM 의 v ↔ h 와 같은 bipartite topology.

### 구조적 등가성

| Harmonium / RBM | GraphLM `HybridGraphLinear` |
|---|---|
| visible `v` ∈ {0,1}^V | input channel `x` ∈ R^{in_features} |
| hidden `h` ∈ {0,1}^H | output channel `y` ∈ R^{out_features} |
| weight `W` ∈ R^{V×H}, bipartite | weight `W` 와 dual adj — bipartite directed |
| `P(h | v)` factorizes (conditional independence) | forward 가 explicit `y = f(W, x)` |
| `P(v, h) ∝ exp(-E)` | (deterministic, energy 직접 정의 없음) |

- **공통점**: bipartite topology 라 inter-layer routing 만 학습. paradigm 의 layer-by-layer Transformer block 도 같은 패턴.
- **차이점**: RBM 은 undirected joint distribution, paradigm 은 directed conditional mapping.

### 차용 가능 아이디어

- **Conditional independence trick**: RBM 의 layer-wise parallelism 은 modern Transformer 의 layer-by-layer forward 와 본질적으로 같은 효율성 패턴.
- **Energy 관점**: paradigm 의 forward `y = (adj_outer · adj_inner · W) · x` 를 energy 관점으로 재해석하면 — *low-energy state = high prediction probability* 의 해석 framework 가 RBM 으로부터 유도 가능.

### Phase 15 prune 의 RBM 해석

- Phase 15 의 edge prune 은 RBM 에서 특정 `(v_i, h_j)` connection 의 `w_ij → 0` 과 등가.
- RBM 학습 후 작은 `|w_ij|` edge 는 contribution 미미 → "자연 sparsity 등장 가능" 의 historical 증거 (DBN 연구에서 확인됨).
- 본 paradigm 의 *30% almost-free prune 구간* 은 RBM 의 low-magnitude weight 자연 발생과 mathematical 일치.

### 후속 실험 가설

- RBM 의 pre-training 패러다임 (DBN stack 학습) 처럼, paradigm 에서도 *layer-wise progressive growth + prune* 이 효과적일 가능성 — Phase 16 후보 (RigL/Net2Net) 의 historical foundation.

## 참고 / 인용

- 공식 코드: 1986 paper, 코드 없음 (이론적 framework)
- 관련 논문:
  - [BM 원조 (Hinton+Sejnowski 1985)](1985-boltzmann-machine-hinton.md)
  - [CD-1 (Hinton 2002)](2002-contrastive-divergence-hinton.md) — RBM 실용 학습
  - [DBN (Hinton+Osindero+Teh 2006)](2006-dbn-hinton.md) — RBM stack
- 본 프로젝트 내 인용 위치: paradigm bipartite reference (향후)
