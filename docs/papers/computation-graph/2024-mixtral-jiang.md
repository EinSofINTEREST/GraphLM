---
title: "Mixtral of Experts"
authors: "Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., et al. (Mistral AI)"
year: 2024
venue: "arXiv preprint"
url: "https://arxiv.org/abs/2401.04088"
arxiv_id: "2401.04088"
code_url: "https://github.com/mistralai/mistral-src"
tags: ["computation-graph", "mixtral", "moe", "llm", "decoder", "sparse", "open-weight"]
status: "draft"
cited_in: []
---

# Mixtral 8×7B — Modern Open-weight MoE LLM

## TL;DR (3줄)

- Mistral 7B 기반의 **decoder Transformer** 의 FFN 을 **8 expert MoE (top-2)** 로 교체한 47B 파라미터 (active 13B) 모델 — apache 2.0 open weights.
- 대부분 dense Llama2-70B 와 비교해서 추론 quality 유사하면서 active param 1/5 → **inference throughput 큰 폭 우위**.
- 본 프로젝트의 **모던 MoE LLM reference** — 학습/추론/serving 의 production-grade end-to-end 사례.

## 핵심 기여

- **8 expert, top-2 routing**: GShard 의 top-2 + 단순화된 capacity. expert 수 적게 (8) 유지해 GPU 메모리 친화적 (Apple Silicon, single-node 8×A100 추론 가능).
- **Sliding-window attention (SWA) + grouped query attention (GQA)** 와 결합 — efficient long-context.
- **Apache 2.0 open-weight** — 학계 / 산업의 MoE 연구를 가속한 가장 영향력 있는 release 중 하나.
- **Routing 의 도메인 specialization 없음** 보고: expert 가 \"코드/수학/일반\" 으로 분리되지 않고, syntactic level (token type) 로 specialize → 직관적 해석과 다름.

## 방법 요약

- 데이터: 비공개 multilingual web corpus + code.
- 모델: 32-layer decoder, hidden 4096, head 32. 각 FFN 이 8 expert (각 14336 dim) MoE layer. 총 47B parameters, active 13B per token.
- 학습: standard causal LM. detail 비공개.
- 핵심 흐름 (각 layer 의 MoE):

$$
y_t = \sum_{i \in \text{Top-2}(G(x_t))} \frac{G(x_t)_i}{\sum_j G(x_t)_j} \cdot E_i(x_t)
$$

top-2 의 weight 를 renormalize 후 weighted sum.

## 실험 / 결과

- MMLU 70.6 vs Llama2-70B 69.8 (Mixtral 이 active 1/5 로 동급).
- HumanEval (코드) 40.2 vs Llama2-70B 29.3 — 큰 폭 우위.
- 다국어 (FR/DE/ES/IT) 모두 Llama2-70B 동급.
- Inference throughput: 동일 메모리에서 dense 70B 보다 4-7x.
- 재현성: 공식 가중치 (Hugging Face), 학습 코드는 비공개. transformers / vllm / llama.cpp 모두 지원.

## 한계 / 비판적 시각

- 학습 데이터 / hyperparameter 비공개 — 재현 연구 어려움.
- 8 expert 는 GShard / Switch (수백~수천) 대비 적음 — \"진짜 sparse 의 효과\" 라기보다 \"compute-efficient ensemble\" 측면.
- Expert 의 token-level specialization 이 비직관적 (도메인 분리 X) — interpretability 한계.
- 모든 expert 를 메모리에 상주시켜야 → 47B 메모리 부담은 dense 와 동일 (active 만 13B 라도).

## 본 프로젝트 시사점

- **현대 MoE LLM 의 sweet spot 참고** — 본 프로젝트가 작은 dataset 의 toy MoE 를 만들 때, expert 수 8~16 정도가 production-quality 의 sweet spot.
- **차용할 아이디어**:
  - **Top-2 with renormalization** — Switch (top-1) 보다 표현력, GShard (top-2 unnormalized) 보다 안정적.
  - **MoE + GQA + SWA 결합** — 작은 모델에서도 효율 트릭의 누적 효과 측정 가치.
  - **\"적은 expert, 큰 expert\" 정책** — Switch 류 (수백 작은 expert) 대비 GPU 친화적.
- **채택하지 않을 부분**: production-scale 학습 detail (비공개) — 본 프로젝트는 toy ~ small scale.
- **후속 실험 가설**:
  - Mixtral 처럼 \"적은 수의 큰 expert\" vs Switch 처럼 \"많은 수의 작은 expert\" — 동일 active param 에서 작은 dataset 의 quality 차이.
  - Expert specialization 의 token-level 패턴이 본 프로젝트의 작은 LM 에서도 재현되는지 (probe 실험).

## 참고 / 인용

- 공식 코드: <https://github.com/mistralai/mistral-src> (inference), 가중치는 HuggingFace `mistralai/Mixtral-8x7B-v0.1`
- 관련 논문: [Switch Transformer](2022-switch-transformer-fedus.md) (top-1), [GShard](2021-gshard-lepikhin.md) (top-2), [Sparsely-Gated MoE](2017-moe-shazeer.md) (조상)
- 본 프로젝트 내 인용 위치: 추후 모던 MoE LLM 의 직접 호출 / fine-tune 실험에서
