---
title: "Training Products of Experts by Minimizing Contrastive Divergence"
authors: "Hinton, G. E."
year: 2002
venue: "Neural Computation, 14(8): 1771-1800"
url: "https://www.cs.toronto.edu/~hinton/absps/nccd.pdf"
doi: "10.1162/089976602760128018"
tags: ["energy-based", "contrastive-divergence", "rbm", "product-of-experts", "gibbs-sampling"]
status: "draft"
cited_in: []
---

# Contrastive Divergence (CD-1) — RBM 실용 학습 breakthrough

## TL;DR (3줄)

- Boltzmann machine 학습의 계산 병목 (negative phase 의 long-run Gibbs sampling) 을 **k step 만의 짧은 Markov chain** 으로 근사하는 contrastive divergence (CD-k) 제안.
- "Product of experts" framework — 여러 expert 분포의 곱을 normalization 하는 모델 학습. RBM 이 핵심 사례.
- CD-1 (단 1 step!) 으로도 RBM 학습이 실용적 수준으로 작동 → 1986 RBM 의 17 년 휴면을 깨운 catalyst, 후일 DBN/DBM 가능하게 함.

## 핵심 기여

- **Contrastive divergence loss**: `CD_k = KL(P_0 || P_∞) - KL(P_k || P_∞)` ≈ `⟨log p(v)⟩_data - ⟨log p(v)⟩_k step` — true gradient 의 biased 근사이지만 작동.
- **CD-1 의 surprising efficacy**: `k=1` (data → 1 Gibbs step) 만으로도 의미 있는 학습 가능 → 계산 비용 huge 감소.
- **Product of experts (PoE)**: `P(v) ∝ ∏_m p_m(v)` 형태의 모델 일반화 — RBM 은 각 hidden unit 이 expert.
- **RBM 학습 recipe 정착**: 후일 모든 DBN/DBM/RBM 연구의 표준 학습 알고리즘으로 등극.

## 방법 요약

- 데이터: handwritten digits, natural images, language modeling 등 다양한 적용
- 모델: RBM (visible-hidden bipartite) — 학습 대상
- 학습 (CD-1):
  1. **Positive phase**: data `v_0` 클램프 → hidden activation `h_0 ~ P(h|v_0)` 샘플링 → `⟨v_0 h_0^\top⟩` 계산
  2. **Negative phase (k=1)**: hidden `h_0` 로부터 visible 재구성 `v_1 ~ P(v|h_0)` → 다시 hidden `h_1 ~ P(h|v_1)` → `⟨v_1 h_1^\top⟩` 계산
  3. **Weight update**: `ΔW = η (⟨v_0 h_0^\top⟩ - ⟨v_1 h_1^\top⟩)`
- 핵심 수식:
  $$ \Delta w_{ij} \approx \eta \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{k\text{-step}} \right) $$

## 실험 / 결과

- 벤치마크: MNIST digit generation, image patches, document modeling
- 주요 수치: 기존 BM 의 모방 불가능했던 큰 RBM 학습이 처음 가능해짐 (수천 unit scale)
- CD-1 의 bias 분석: theoretically biased 이지만 실용적으로 작동 — Carreira-Perpinan & Hinton 2005 후속 분석으로 확인

## 한계 / 비판적 시각

- **CD-k 의 theoretical bias**: true gradient 가 아니라 근사 — 학습이 항상 likelihood 를 증가시키지 않을 수 있음
- **convergence 의 정성적 보장 부재**: 실용적 작동은 입증되었으나 분석적 guarantee 없음
- **modern deep learning 과의 거리**: backprop 기반 end-to-end 학습이 표준이 된 후 RBM/CD 의 직접 사용 감소
- 그러나 historical 의의는 명확 — 2006 DBN 의 기술적 가능성 제공

## 본 프로젝트 시사점

> Paradigm 의 **gradient estimation efficiency** 측면의 historical 시조.

### CD-1 의 "biased but practical" 철학

- CD-1 의 본질 = *완벽한 gradient 대신 짧은 sampling 으로 적당한 근사 → 실용성 우선*
- Paradigm 의 Phase 15 prune 도 비슷한 철학: *정확한 importance 측정 대신 magnitude 라는 proxy 사용*
- "Biased but practical" 의 가치 — perfectionism 보다 작동하는 simple rule.

### graph energy + gradient 연결

| Contrastive Divergence | GraphLM paradigm |
|---|---|
| `⟨v_i h_j⟩_data - ⟨v_i h_j⟩_model` (statistic difference) | `∂L/∂W` via backprop (gradient) |
| Markov chain 으로 model statistics 근사 | 단일 forward + backward 로 gradient 정확 계산 |
| edge importance 추정 (positive vs negative phase 차이) | edge importance 추정 (gradient magnitude 또는 weight magnitude) |

- **공통점**: edge importance 를 *통계적 차이* 로 측정.
- **차이점**: CD 는 sampling, paradigm 은 backprop. Modern hardware 에서 backprop 의 비용이 sampling 만큼 가능해진 후 CD 는 niche 로.

### Phase 15 prune 의 CD 관점 재해석

- Phase 15 의 `prune_by_magnitude` 는 *현재 모델의 edge 중요도 = magnitude* 라는 한 가지 metric 만 사용.
- CD 의 *data-driven vs prior-driven 차이* 를 도입하면 — 학습 중 *data 에 반응하는 edge* vs *prior 에서 안정적인 edge* 의 구분 가능.
- **차용할 아이디어**: prune 의 importance metric 으로 magnitude 외 *데이터 의존성* 추가 — 가령 batch 별 gradient 의 variance.

### 채택하지 않을 부분

- Sampling 자체는 paradigm 에 도입하지 않음 (Transformer deterministic forward 와 mismatch, 위 Hinton+Sejnowski 1985 와 같은 이유).
- CD-1 의 학습률 / scheduler 등 RBM-specific recipe 는 비적용.

### 후속 실험 가설

- Phase 15 prune 의 metric 으로 *gradient variance 기반 importance* (= "이 edge 가 batch 마다 얼마나 다른 신호를 받는가") 추가 — CD 의 statistical difference 와 spirit 일치.

## 참고 / 인용

- 공식 코드: Hinton lab 의 RBM toolkit (Matlab, 후일 다양한 framework 로 포팅)
- 관련 논문:
  - [Smolensky 1986 (RBM 원조)](1986-harmony-theory-smolensky.md)
  - [DBN (Hinton+Osindero+Teh 2006)](2006-dbn-hinton.md) — CD 의 stacked 활용
  - [DBM (Salakhutdinov+Hinton 2009)](2009-dbm-salakhutdinov.md) — multi-layer CD
- 본 프로젝트 내 인용 위치: importance metric reference (향후)
