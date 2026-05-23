---
title: "DARTS: Differentiable Architecture Search"
authors: "Liu, H., Simonyan, K., & Yang, Y."
year: 2019
venue: "ICLR 2019"
url: "https://arxiv.org/abs/1806.09055"
arxiv_id: "1806.09055"
code_url: "https://github.com/quark0/darts"
tags: ["computation-graph", "dynamic-param", "darts", "nas", "differentiable", "gradient-based", "architecture-search"]
status: "draft"
cited_in: []
---

# DARTS — Differentiable Architecture Search

## TL;DR (3줄)

- Architecture search 의 candidate operation 집합을 **continuous relaxation** (softmax 가중 합) 으로 표현 → gradient descent 로 architecture 와 weight 동시 학습.
- 기존 NAS (RL / evolution 기반) 의 수천 GPU-day 비용을 단 **수 GPU-day** 로 단축 — search efficiency 의 mile stone.
- 본 프로젝트 패러다임에서 **\"architecture 가 학습 가능 변수가 되어 최종 파라미터 수가 학습 산물\"** 인 가장 직접적 사례.

## 핵심 기여

- **Continuous relaxation**: 각 edge $(i, j)$ 의 operation 선택을 $\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} o(x)$ 의 softmax 가중 합으로 표현. $\alpha$ 가 architecture 변수.
- **Bilevel optimization**: weight $w$ 와 architecture $\alpha$ 의 nested 최적화 — inner loop 는 $w$, outer loop 는 $\alpha$.
- **One-shot training**: 모든 candidate op 가 supernet 의 일부로 동시 학습. 최종 architecture 는 argmax $\alpha$ 로 discretize.
- **Search cost 1.5 GPU-day** (CIFAR-10): RL 기반 NASNet (2000 GPU-day) 대비 3 orders of magnitude.

## 방법 요약

- 데이터: CIFAR-10 (architecture search) → ImageNet (transfer), Penn Treebank (RNN cell search).
- 모델: cell-based search space — normal cell + reduction cell. 각 cell 은 7-node DAG, 노드 간 edge 마다 8개 candidate operation (Conv 3×3, Conv 5×5, MaxPool, identity 등).
- 학습:
  1. Bilevel: weight $w$ 학습 (train set) + architecture $\alpha$ 학습 (val set)
  2. 수렴 후 각 edge 마다 argmax $\alpha$ 로 single operation 선택 → final architecture
  3. 선택된 architecture 를 처음부터 재학습
- 핵심 수식 (bilevel, 원논문 §2.2 Eq. 3):

$$
\min_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \quad \text{s.t.}\quad w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)
$$

second-order approximation 으로 $\nabla_\alpha \mathcal{L}_{\text{val}}$ 계산.

## 실험 / 결과

- CIFAR-10 test error 2.76% (cell-based SOTA at the time).
- ImageNet (transferred cell) top-1 73.1%.
- Penn Treebank (RNN) test perplexity 56.1.
- Search cost: 1.5 GPU-day vs NASNet 의 2000 GPU-day.
- 재현성: 공식 PyTorch 공개. 후속 작업이 비교 baseline 으로 광범위 채택.

## 한계 / 비판적 시각

- **Skip-connection collapse** — 학습이 진행되면서 모든 edge 가 zero-cost operation (skip-connection) 으로 수렴하는 현상 보고 (P-DARTS, R-DARTS 가 보완).
- **Bilevel gradient approximation** 의 불안정 — second-order 가 정확하지만 비용 높음, first-order approximation 은 quality 손실.
- Search 와 evaluation 단계의 **architecture mismatch** — search 의 supernet 환경과 final architecture 환경이 달라 transfer 시 quality drop.
- Memory 부담 — supernet 이 모든 candidate op 를 동시에 들고 있어야 → 큰 search space 에 비현실적.

## 본 프로젝트 시사점

- **\"Architecture 가 학습 가능 변수\" 의 가장 직접적 사례** — 본 프로젝트가 \"파라미터 수를 학습 결과로 결정하고 싶다\" 면 DARTS 의 continuous relaxation 이 기본 recipe.
- **차용할 아이디어**:
  - **Softmax-weighted operation mixing** — 본 프로젝트의 \"이 layer 가 있을지 없을지\" 결정을 continuous 로 표현해 gradient 학습 가능하게.
  - **Bilevel separation** of weight vs architecture — 본 프로젝트의 학습 loop 가 두 optimizer 를 분리해 운영할 reference.
  - **Discretize at end** — supernet 학습 후 argmax 로 final structure 추출 패턴.
- **채택하지 않을 부분**: vision-specific cell-based search space — 본 프로젝트는 Transformer block 수준의 더 거시적 search.
- **후속 실험 가설**:
  - DARTS 의 skip-collapse 가 Transformer block 의 \"이 layer 가 있을지\" 선택에서도 발생하는지 — 발생 시 P-DARTS 류 \"점진적으로 zero op 제거\" 채용 검토.
  - $\alpha$ 의 temperature schedule (search 후반 hardening) 이 final architecture 의 quality 에 미치는 영향.

## 참고 / 인용

- 공식 코드: <https://github.com/quark0/darts> (PyTorch)
- 관련 논문: [AutoFormer](2021-autoformer-chen.md) (Transformer NAS, supernet 방식), [GHN-3](2023-ghn3-knyazev.md) (architecture-as-graph 의 다른 접근)
- 본 프로젝트 내 인용 위치: 추후 architecture search 실험 노트북에서
