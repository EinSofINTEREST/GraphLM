---
title: "Masked Structural Growth for 2x Faster Language Model Pre-training"
authors: "Yao, Y., Zhang, Z., Li, X., & Sun, M."
year: 2024
venue: "ICLR 2024"
url: "https://arxiv.org/abs/2305.02869"
arxiv_id: "2305.02869"
code_url: "https://github.com/cofe-ai/MSG"
tags: ["computation-graph", "dynamic-param", "msg", "masked-growth", "llm", "depth-width", "function-preserving"]
status: "draft"
cited_in: []
---

# MSG — Masked Structural Growth for LLM Pre-training

## TL;DR (3줄)

- 큰 LLM 의 전체 architecture 를 미리 설계한 뒤, **mask 로 일부를 가린 작은 모델로 시작 → 학습 중 mask 를 점진적으로 unmask** 해 큰 모델로 성장.
- Depth (layer) + width (hidden / FFN / head) 모두 strict function-preserving 으로 unmask — bert2BERT / LiGO 의 한계 (depth-only 또는 specific pair) 극복.
- LLM 사전학습 시간 **2x 단축** 보고 — GPT 류 (decoder) 에도 안정 적용. 본 프로젝트의 **LLM-scale growing reference**.

## 핵심 기여

- **Strict function-preserving growth for all 4 axes**: depth (layer 수), width (hidden dim), FFN intermediate dim, attention head 수 — 각각의 unmask schedule 이 모두 strict preservation 보장.
- **Masked architecture initialization**: 처음부터 large architecture 의 weight tensor 가 존재하되 대부분 mask 처리 → 학습은 unmask 된 부분만.
- **Continuous unmask schedule**: 학습 step 마다 mask 의 일부를 unmask — 점진적 capacity 증가.
- **LLM decoder 적용**: GPT-style autoregressive LM 에서 안정 학습 검증 — 이전 growing 작업들이 encoder (BERT) 위주였던 한계 보완.

## 방법 요약

- 데이터: C4 (영어 web corpus), Pile (LLM 표준).
- 모델: GPT-2 family (124M → 774M), BERT family (검증).
- 학습:
  1. Target large architecture 의 weight tensor 전체 allocate
  2. Mask 로 작은 effective architecture 만 활성화 (예: 4-layer / 256 hidden 으로 시작)
  3. Pretraining step 진행하며 schedule 에 따라 mask 의 일부 unmask
  4. Final step 에서 전체 unmask (= target large architecture)
- 핵심 흐름 (depth unmask 예시):

각 layer 의 residual 출력에 masked scale 곱:
$$
h_\text{out} = h_\text{in} + s_l \cdot \text{Block}_l(h_\text{in}), \quad s_l \in [0, 1]
$$

$s_l = 0$ 이면 layer skip (function-preserving), schedule 따라 $s_l \to 1$.

## 실험 / 결과

- GPT-2 (124M → 350M / 774M): 처음부터 학습 대비 **wallclock ~2x 단축** (동일 perplexity 도달).
- bert2BERT / LiGO 대비도 step 효율 우위 — LLM 의 큰 batch / 긴 학습에서 효과 누적.
- Downstream zero-shot (LAMBADA, HellaSwag): 동등 또는 약간 우위.
- 학습 안정성: function preservation 으로 unmask 직후 perplexity spike 없음.
- 재현성: 공식 PyTorch 공개.

## 한계 / 비판적 시각

- Target architecture 를 미리 정해야 함 — \"학습 중 final size 도 동적 결정\" 까지는 미해결 (DARTS 류와의 결합 여지).
- Mask 자체의 메모리 — 모든 weight tensor 가 처음부터 allocate → 메모리는 large model 수준 (compute 만 small).
- Schedule (언제 어느 mask 를 unmask) 의 hyperparameter — 자동 최적화는 future work.
- 검증이 ~1B 까지 — 진짜 LLM-scale (7B+) 효과 미보고.

## 본 프로젝트 시사점

- **LLM-scale growing 의 modern reference** — 본 프로젝트가 작은 GraphLM 으로 시작해 큰 모델로 확장하는 framework 의 가장 최신 recipe.
- **차용할 아이디어**:
  - **Masked-then-unmask** 패턴 — 메모리 비용을 들여 strict function preservation 보장하는 trade-off. 본 프로젝트의 toy 실험 단계에선 메모리 부담 작아 적용 용이.
  - **4 axes simultaneous growth** (depth + width + FFN + heads) — bert2BERT/LiGO 의 axis 중 일부만 다루는 한계 극복.
  - **Decoder LM 검증** — 본 프로젝트가 autoregressive 방향이라면 BERT-only 작업들보다 직접적 reference.
- **채택하지 않을 부분**: full target architecture 의 메모리 사전 allocation — 정말 큰 모델에는 별도 sharding 필요.
- **후속 실험 가설**:
  - Unmask schedule 의 \"linear vs cosine vs adaptive\" — quality / 학습 안정성에 미치는 영향.
  - MSG 의 4 axes 중 어느 축의 expansion 이 quality 에 가장 기여하는지 ablation — 본 프로젝트가 어떤 축에 집중할지 결정 근거.

## 참고 / 인용

- 공식 코드: <https://github.com/cofe-ai/MSG> (PyTorch)
- 관련 논문: [Net2Net](2016-net2net-chen.md) (origin), [Progressive Stacking](2019-stacking-gong.md) (depth-only growth), [bert2BERT](2022-bert2bert-chen.md) (BERT pair), [LiGO](2023-ligo-wang.md) (learnable operator)
- 본 프로젝트 내 인용 위치: 추후 LLM-scale growing 실험 노트북에서
