---
title: "A Fast Learning Algorithm for Deep Belief Nets"
authors: "Hinton, G. E., Osindero, S., & Teh, Y. W."
year: 2006
venue: "Neural Computation, 18(7): 1527-1554"
url: "https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf"
doi: "10.1162/neco.2006.18.7.1527"
tags: ["energy-based", "dbn", "deep-learning-resurrection", "layer-wise-pretraining", "stacked-rbm"]
status: "draft"
cited_in: []
---

# Deep Belief Network (DBN) — 딥러닝 부활의 시발점

## TL;DR (3줄)

- 여러 RBM 을 **layer-wise greedy pre-training** 후 wake-sleep / fine-tune 으로 결합한 **첫 실용적 deep generative network**.
- "왜 deep network 가 학습되지 않는가" (vanishing gradient, local minima) 문제를 *unsupervised layer-wise pre-training* 으로 우회.
- **딥러닝 부활** (2006-2012 era) 의 catalyst — MNIST SOTA 갱신, ICML/NIPS deep learning 흐름 촉발.

## 핵심 기여

- **Layer-wise greedy pre-training**:
  1. 첫 RBM (visible-hidden₁) 학습 (CD-1)
  2. hidden₁ 의 expectation 을 새 "data" 로 사용해 두 번째 RBM (hidden₁-hidden₂) 학습
  3. 반복 → N-layer DBN 완성
- **Variational bound 증명**: 새 layer 추가가 data log-likelihood 의 variational bound 를 (변경 안 함 OR 향상시킴) → "더 깊게 쌓아도 안 해롭다" 보장.
- **Wake-sleep fine-tuning**: pre-training 후 generative / recognition weight 를 contrastive sleep wake update 로 fine-tune.
- **Discriminative variant**: pre-trained DBN 위에 softmax classifier 얹어 supervised fine-tune → MNIST 1.25% error 달성 (당시 SOTA).

## 방법 요약

- 데이터: MNIST digit images (대표적)
- 모델: 3-layer DBN (784-500-500-2000 또는 유사 구성)
- 학습 phase:
  - **Pre-training**: 각 RBM 을 CD-1 으로 greedy 학습 (~30-50 epochs / layer)
  - **Fine-tuning**: wake-sleep (generative model) 또는 backprop (discriminative classifier)
- 핵심 수식:
  - DBN 의 joint distribution: `P(v, h¹, h², ..., hᴸ) = P(hᴸ⁻¹, hᴸ) ∏ₗ P(hˡ⁻¹ | hˡ)`
  - 최상위 두 layer 만 undirected (RBM), 나머지는 directed top-down

## 실험 / 결과

- 벤치마크: MNIST (digit classification + generation)
- 주요 수치:
  - Generative model: log P(v) 가 기존 deep network 대비 우수
  - Discriminative: MNIST 1.25% error (당시 SOTA 갱신)
  - Generation 품질: 학습 안 된 영역도 합리적 digit 생성 가능
- 재현성 메모: Hinton lab 의 Matlab 코드 공개, 후일 다양한 deep learning framework 로 포팅

## 한계 / 비판적 시각

- **2012 이후 가치 감소**: ReLU + dropout + 좋은 initialization 등이 pre-training 없이도 deep network 학습 가능하게 함 → DBN pre-training 의 실용적 가치 감소
- **순수 supervised setting 에서는 backprop end-to-end 가 더 강력**: 충분한 labeled data 있으면 pre-training 불필요
- **modern Transformer 는 DBN 과 다른 길**: 그러나 그 *layer-wise progressive* 개념은 후일 LiGO/MSG (computation-graph 카테고리) 로 재발견됨
- 그러나 *unsupervised pre-training + supervised fine-tune* 2-stage 패러다임은 후일 BERT/GPT 시대에 부활

## 본 프로젝트 시사점

> Paradigm 의 **layer-wise progressive growth** 관점에서 *historical precursor*. 또한 *energy-based deep generative* 의 paradigm 비교.

### Layer-wise growth 의 historical 의의

| DBN pre-training | GraphLM Phase 16 후보 |
|---|---|
| RBM 1 학습 → freeze → RBM 2 학습 → ... | Net2Net/LiGO 식 layer 추가 후 학습 |
| 각 layer 가 lower-layer 의 latent 표현을 새로 학습 | growth 후 신규 layer 가 기존 representation 보존하며 학습 |
| function preservation 보장 (variational bound) | function preservation 보장 (Net2Net 의 핵심) |
| greedy → 후속 fine-tune | growth → continued training |

- **공통점**: 모델을 한 번에 학습 안 하고 **점진적 깊이/너비 확장 + 단계별 학습** 이라는 핵심 idea.
- **차이점**: DBN 은 generative + unsupervised, modern growth (Net2Net/LiGO) 는 discriminative + supervised. DBN 의 pre-training 가치는 LLM era 의 BERT/GPT 의 unsupervised pre-training 으로 *재발견* 됨.

### Phase 16 (Net2Net/LiGO) 의 historical roots

- Phase 16 후보의 **function-preserving growth** 는 DBN 의 *layer 추가가 variational bound 를 깨지 않는다* 와 동일한 spirit.
- DBN 은 "더 깊게 쌓아도 worse 아님" 의 첫 deep generative 증명. paradigm 의 Phase 16+ 의 *grow + shrink 로 capacity 동적 조정* 도 같은 가정에 의존.

### Energy + deep 의 관계

- DBN 의 top 2 layer 만 RBM (energy-based), 아래는 directed → **하이브리드 energy/directed graph**.
- Paradigm 의 `HybridGraphLinear` 도 *energy formulation 직접 사용 안 함* 이지만, *graph edge weight 학습* 이라는 구조는 RBM 의 directed analog.
- DBN 의 hybrid 구조가 paradigm 의 "*표면은 deterministic forward, 내부는 graph energy-like topology*" 와 유사.

### 차용 가능 아이디어

- **Layer-wise progressive training**: paradigm 의 *layer 별 차등 prune/grow* (Phase 17 후보) 의 historical basis.
- **Variational bound 기반 growth 결정**: paradigm 에서도 *growth criterion 으로 bound improvement* 사용 가능 — gradient norm 등.

### 채택하지 않을 부분

- Generative pre-training 자체는 paradigm 의 1순위 아님 (LM 의 unsupervised pre-training 은 이미 standard).
- DBN 의 wake-sleep algorithm 은 modern backprop 대비 비효율.

### 후속 실험 가설

- **Layer-wise Phase 15 prune**: 전체 model 일괄 prune 대신 *bottom-up layer-by-layer* prune — DBN 의 greedy 패러다임을 paradigm 의 dynamic shrink 에 적용.

## 참고 / 인용

- 공식 코드: Hinton lab Matlab toolkit, 후일 PyTorch/TF 포팅 다수
- 관련 논문:
  - [CD-1 (Hinton 2002)](2002-contrastive-divergence-hinton.md) — DBN 의 building block
  - [DBM (Salakhutdinov+Hinton 2009)](2009-dbm-salakhutdinov.md) — 완전 undirected 다층 확장
  - [Net2Net (Chen 2016)](../computation-graph/2016-net2net-chen.md) — function-preserving growth 의 modern 형태
  - [LiGO (Wang 2023)](../computation-graph/2023-ligo-wang.md) — learnable expansion
- 본 프로젝트 내 인용 위치: layer-wise growth reference (향후)
