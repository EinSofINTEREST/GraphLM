---
title: "Deep Boltzmann Machines"
authors: "Salakhutdinov, R., & Hinton, G. E."
year: 2009
venue: "AISTATS 2009"
url: "https://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf"
tags: ["energy-based", "dbm", "deep-boltzmann", "undirected", "bidirectional-coupling"]
status: "draft"
cited_in: []
---

# Deep Boltzmann Machine (DBM) — 완전 undirected 다층 Boltzmann

## TL;DR (3줄)

- DBN 의 hybrid 구조 (top 만 undirected) 와 달리 **모든 layer 가 RBM-style undirected** 인 진정한 deep Boltzmann machine.
- 양방향 정보 전파 (bottom-up + top-down 동시) 로 더 풍부한 internal representation 가능, 학습은 *layer-wise pre-training + mean-field joint training* 의 2-stage.
- MNIST/NORB SOTA 갱신, 후속 generative 연구에 영향. 그러나 학습 복잡도와 sampling cost 때문에 2014+ era 의 GAN/VAE 에 밀림.

## 핵심 기여

- **All-layer undirected**: DBN 의 directed lower layer 들도 RBM-style 으로 일괄 undirected — 정보 전파가 bidirectional.
- **Inference 의 도전과 해결**: 다층 undirected 는 exact inference 불가 → mean-field variational approximation 으로 hidden state estimation.
- **Greedy + joint training**:
  1. DBN 처럼 layer-wise RBM pre-training (단, 1번째와 마지막 RBM 의 weight 를 2배로 — bidirectional coupling 보정)
  2. 전체 model 을 mean-field + persistent CD 로 joint fine-tune
- **표현력 우위 입증**: MNIST 0.95% error (당시 DBN 1.25% 대비 개선), NORB benchmark SOTA

## 방법 요약

- 데이터: MNIST, NORB (3D object), CIFAR (선택적)
- 모델: 3-layer DBM (가령 784-500-500-2000)
- 학습:
  - **Pre-training**: 각 RBM 을 CD-1 으로 학습. 단 1번째 RBM (visible-h¹) 과 마지막 RBM (h^{L-1}-h^L) 의 weight 를 2x — joint phase 에서의 bidirectional input 보정.
  - **Joint fine-tuning**:
    - Mean-field inference: `μⱼ = σ(Σᵢ wᵢⱼ vᵢ + Σₖ wⱼₖ μₖ)` iterative update
    - Persistent CD (PCD): negative samples 를 single chain 으로 유지
    - Weight update: positive (data-driven) - negative (model-driven) statistics
- 핵심 수식:
  $$ E(v, h¹, ..., h^L) = -v^\top W^{(1)} h^{(1)} - \sum_{l=1}^{L-1} h^{(l)\top} W^{(l+1)} h^{(l+1)} $$

## 실험 / 결과

- 벤치마크: MNIST classification + generation, NORB 3D object
- 주요 수치:
  - MNIST: 0.95% error (당시 SOTA, DBN 1.25% 대비)
  - NORB: 7.2% error (당시 SOTA)
  - Generation quality: DBN 보다 자연스러운 sample
- 재현성 메모: 학습 시간 매우 김 (DBN 대비 ~10x), 공식 Matlab 코드 공개

## 한계 / 비판적 시각

- **학습 복잡도 폭증**: pre-training 의 weight doubling + mean-field iteration + PCD 의 chain 관리 → 매우 brittle
- **convergence 의 sensitivity**: hyperparameter (mean-field iteration 수, PCD chain 길이) 에 매우 민감
- **scaling limitation**: 큰 model (현대 deep learning 기준) 에 사실상 적용 불가
- **GAN (2014) / VAE (2013) 등장 후 사실상 archival**: 같은 generative 문제를 더 효율적으로 푸는 방법 등장
- DBM 자체는 deprecated 이나 *bidirectional + multi-layer + undirected* 의 conceptual contribution 은 modern Transformer (bidirectional attention) 와 spirit 일치

## 본 프로젝트 시사점

> Paradigm 의 **bidirectional coupling + multi-layer graph energy** 의 가장 정교한 historical example. 비록 학습 방법은 deprecated 이지만 *모든 layer 가 graph energy 단위* 라는 구조가 paradigm 과 직접 정렬.

### 모든 layer 가 graph 인 구조

| Deep Boltzmann Machine | GraphLM Phase 14+ |
|---|---|
| 모든 layer 가 RBM (undirected) | 모든 block 이 `HybridGraphLinear` (Phase 14 full graph block) |
| bidirectional coupling (`P(h¹|v, h²)` 양쪽 의존) | unidirectional residual stream (forward 만) |
| Mean-field inference 의 multi-step | single forward pass |
| layer-wise pre-training + joint fine-tune | end-to-end training (현재) — 단, Phase 16 후보의 *progressive growth* 는 layer-wise spirit 차용 |

- **공통점**: *모든 layer 가 graph 표현 단위* 라는 구조 — Phase 14 의 full graph block 이 DBM 의 multi-layer RBM 구조와 conceptual 등가.
- **차이점**: DBM 은 undirected + sampling, paradigm 은 directed + deterministic.

### Phase 17 후보 (layer-wise 차등 prune) 의 DBM 관점

- DBM 의 layer-wise pre-training 은 *layer 마다 다른 학습 동역학* 을 인정하는 접근.
- Phase 17 후보 (attention vs FFN 별 다른 sparsity target) 도 같은 spirit — layer 마다 *capacity 필요량 다름* 가정.
- DBM 의 *first/last RBM 의 weight doubling* (bidirectional boundary 보정) 은 paradigm 에서 *boundary layer (embedding, lm_head) 의 special handling* 영감 제공.

### Inference complexity 의 교훈

- DBM 의 mean-field inference 가 multi-step 인 이유 = bidirectional coupling 의 inference 비용.
- Paradigm 이 *deterministic single forward* 를 유지하는 한 이 cost 회피 — DBM 의 lesson 은 *bidirectional 의 매력 vs 계산 비용 trade-off*.

### 차용 가능 아이디어

- **Layer-wise differentiation**: 같은 graph 구조라도 layer 마다 *서로 다른 prune/grow 정책* 적용 — Phase 17 의 motivation.
- **Boundary layer 특수 처리**: paradigm 의 embedding 과 lm_head 는 hybrid linear 가 아님. DBM 의 1번째/마지막 layer 특수 보정과 같은 boundary handling 의 정당성.

### 채택하지 않을 부분

- Undirected + sampling 자체는 paradigm 에 도입 X (Transformer deterministic).
- Mean-field iteration 의 multi-step inference 도 비채택 (계산 비용).
- PCD 등 RBM-specific 학습 trick 도 비적용.

### 후속 실험 가설

- **Phase 17a (layer-wise prune)**: DBM 의 layer-wise spirit 를 차용해 *block 0 vs block N* 의 sparsity target 차등화. 가설: 깊은 layer 가 더 sparse 가능 (DBM 의 top RBM 이 더 abstract 표현).
- **Phase 17b (boundary handling)**: embedding/lm_head 의 prune 정책을 hybrid block 과 차등화 — DBM 의 boundary doubling 처럼 special treatment.

## 참고 / 인용

- 공식 코드: Salakhutdinov 의 Matlab DBM toolkit (Toronto)
- 관련 논문:
  - [DBN (Hinton+Osindero+Teh 2006)](2006-dbn-hinton.md) — DBM 의 직전 단계
  - [CD-1 (Hinton 2002)](2002-contrastive-divergence-hinton.md) — building block
  - [BM 원조 (Ackley+Hinton+Sejnowski 1985)](1985-boltzmann-machine-hinton.md) — root
  - [LiGO (Wang 2023)](../computation-graph/2023-ligo-wang.md) — DBM 의 layer-wise spirit 의 modern 형태
- 본 프로젝트 내 인용 위치: layer-wise differentiation reference (Phase 17 후보, 향후)
