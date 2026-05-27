---
title: "A Learning Algorithm for Boltzmann Machines"
authors: "Ackley, D. H., Hinton, G. E., & Sejnowski, T. J."
year: 1985
venue: "Cognitive Science, 9(1): 147-169"
url: "https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf"
doi: "10.1207/s15516709cog0901_7"
tags: ["energy-based", "boltzmann-machine", "stochastic", "graph-energy", "gibbs-sampling"]
status: "draft"
cited_in: []
---

# Boltzmann Machine (BM) — 원조 energy-based stochastic network

## TL;DR (3줄)

- 노드가 binary stochastic unit 인 fully-connected graph 위에서 energy minimization 으로 학습하는 generative model — Hopfield network 의 stochastic 확장.
- Simulated annealing 으로 visible/hidden unit 의 결합 분포를 학습, Gibbs sampling 기반 contrastive update rule (positive/negative phase).
- 학습 신호 = data-clamped state 와 free-running state 의 통계 차이. 1980 년대 neural network 의 graph energy 학습 방법론 확립.

## 핵심 기여

- **Stochastic neural network 정식화**: deterministic Hopfield (1982) 의 일반화. unit 이 `s_i ∈ {0, 1}` 인데 결정값이 아니라 확률적 (`P(s_i = 1) = σ(net_i / T)`).
- **Energy function**: `E(s) = -Σ_{i<j} w_ij s_i s_j - Σ_i b_i s_i`. graph 의 edge weight `w_ij` 가 학습 대상.
- **Boltzmann distribution**: `P(s) ∝ exp(-E(s) / T)` — 학습 목표가 데이터의 marginal 분포를 모델 분포에 매칭.
- **Contrastive learning rule**: `Δw_ij ∝ ⟨s_i s_j⟩_data - ⟨s_i s_j⟩_model` — positive phase (data clamped) vs negative phase (free running) 의 차이.
- **Hidden units 도입** — visible 만 있던 Hopfield 대비 표현력 확장.

## 방법 요약

- 데이터: binary patterns (당시 toy problems — encoder/decoder, parity 등)
- 모델: fully-connected undirected graph (`N` units, weights symmetric `w_ij = w_ji`)
- 학습:
  - **Simulated annealing** 으로 temperature `T` 를 점진적으로 낮추며 thermal equilibrium 도달
  - Positive phase: visible unit clamped to data, hidden unit 들이 equilibrium 으로 수렴 → `⟨s_i s_j⟩_data` 측정
  - Negative phase: 모든 unit free, equilibrium 도달 → `⟨s_i s_j⟩_model` 측정
  - Weight update: `Δw_ij = η (⟨s_i s_j⟩_data - ⟨s_i s_j⟩_model)`
- 핵심 수식:
  $$ E(s) = -\sum_{i<j} w_{ij} s_i s_j - \sum_i b_i s_i $$
  $$ P(s) = \frac{\exp(-E(s)/T)}{Z}, \quad Z = \sum_{s'} \exp(-E(s')/T) $$

## 실험 / 결과

- 벤치마크: encoder/decoder task (4-2-4, 8-3-8 patterns), shifter task, parity check
- 주요 수치: 작은 toy 문제에서 hidden unit 활용한 internal representation 학습 성공 입증
- 재현성 메모: 1985 hardware 한계로 모든 결과가 매우 작은 scale. 본 논문의 의의는 framework 자체이지 수치적 SOTA 가 아님

## 한계 / 비판적 시각

- **계산 비용 막대**: 매 weight update 마다 simulated annealing 2 phase 실행 필요 → 사실상 큰 network 에 불가능
- **Equilibrium 가정의 비현실성**: 실제 simulation 에서 true equilibrium 도달 어려움 → 학습 instability
- **fully-connected 라 scale 안 됨**: weight matrix `O(N²)`
- 이 한계들이 17 년 후 Hinton 2002 (Contrastive Divergence) 로 일부 해결, Smolensky 1986 (RBM) 로 connectivity 제약 도입
- **딥러닝 발전 과정에서 한동안 잊혀짐** — 2006 DBN 으로 재부활 (Hinton+Osindero+Teh)

## 본 프로젝트 시사점

> GraphLM 의 paradigm 과 **graph energy 표현의 직접 선조**.

### graph energy 표현의 구조적 유사성

| Boltzmann Machine | GraphLM `HybridGraphLinear` |
|---|---|
| `E = -Σ w_ij s_i s_j` (energy via edge weights) | `eff_w = adj_outer · adj_inner · W` (effective edge weight) |
| graph 의 모든 edge `w_ij` 가 학습 대상 | edge magnitude 가 학습 대상 (adj_outer, adj_inner) |
| symmetric undirected graph | directed bipartite (input → output channels) |
| binary stochastic units | continuous deterministic units (Transformer hidden) |
| Boltzmann distribution 학습 | gradient descent 로 loss 최소화 |

- **공통점**: graph edge weight 가 학습 대상이라는 핵심 — paradigm 의 magnitude rule (≈ 1.0 sweet spot) 의 historical motivation.
- **차이점**: BM 은 stochastic + sampling-based, paradigm 은 deterministic + gradient-based. BM 의 contrastive update 가 explicit `⟨s_i s_j⟩` 통계인 반면, paradigm 의 backprop 은 chain rule 의 implicit 통계 추정.

### dynamic 위상 변화의 BM 관점

- Phase 15 의 edge prune (mask=0) 은 BM 에서 `w_ij → -∞` 시 `s_i, s_j` 간 interaction 차단과 등가.
- BM 의 학습은 energy landscape 의 모양 자체를 학습 — paradigm 의 *topology + magnitude 동시 학습* 의 mathematical precursor.
- **차용할 아이디어**: BM 의 positive/negative phase 의 contrastive 통계 → paradigm 에서 *학습된 connectivity 의 importance metric* 으로 활용 가능 (어떤 edge 가 data-driven activity 와 prior activity 의 차이가 큰가).

### 채택하지 않을 부분

- Sampling-based learning 자체는 paradigm 에 도입하지 않음 — Transformer 의 deterministic forward 와 mismatch.
- Symmetric weight 제약도 부적합 (Transformer 는 asymmetric directed).

### 후속 실험 가설

- BM 의 `⟨s_i s_j⟩_data - ⟨s_i s_j⟩_model` 통계와 유사한 *adj importance metric* 을 정의해 Phase 15 prune 의 magnitude 기준 외 대안 평가 — 학습 동안 활성도 차이가 큰 edge 우선 보존.

## 참고 / 인용

- 공식 코드: 1985 paper, 코드 없음
- 관련 논문:
  - [Smolensky 1986 (RBM 원조)](1986-harmony-theory-smolensky.md)
  - [Hinton 2002 (CD-1)](2002-contrastive-divergence-hinton.md)
  - Hopfield 1982 (deterministic 전신, 외부 reference)
- 본 프로젝트 내 인용 위치: paradigm reference (cited_in 비어있음, 향후 사용 시 갱신)
