---
title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
authors: "Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J."
year: 2017
venue: "ICLR 2017"
url: "https://arxiv.org/abs/1701.06538"
arxiv_id: "1701.06538"
code_url: "https://github.com/davidmrau/mixture-of-experts"
tags: ["computation-graph", "moe", "sparse", "gating", "conditional-computation", "foundational"]
status: "draft"
cited_in: []
---

# Sparsely-Gated MoE — Conditional Computation 의 시초

## TL;DR (3줄)

- 한 layer 에 $N$ 개의 expert 네트워크를 두고, **learnable gating network 가 각 입력마다 top-$k$ expert 만 활성화** 하는 sparse Mixture-of-Experts layer 를 제안.
- 모델 파라미터 수는 137B 까지 키우면서도 inference 비용은 dense 모델의 일부만 — **conditional computation** 의 실용적 첫 증명.
- 본 프로젝트 **computation-as-graph 패러다임의 시초** — token 이 expert 노드들 중 일부를 동적으로 선택하는 routing 학습.

## 핵심 기여

- **Top-$k$ gating with noisy gating**: $G(x) = \text{softmax}(\text{KeepTopK}(H(x), k))$, $H(x) = xW_g + \mathcal{N}(0,1) \cdot \text{softplus}(xW_{noise})$ — sparsity 강제 + 학습 안정성.
- **Auxiliary load-balancing loss**: expert 별 활용도가 한쪽으로 쏠리는 *expert collapse* 방지 — importance + load 의 CV 제곱.
- **LSTM language modeling 에서 137B 파라미터 모델** 학습 성공 — dense 모델 대비 perplexity 큰 폭 개선.

## 방법 요약

- 데이터: 1B Word benchmark, 100B Google News corpus (LM); WMT'14 En-Fr (NMT).
- 모델: LSTM stack 사이에 sparse MoE layer 삽입. expert 는 작은 FFN. $N$ 은 수백~수천.
- 학습: SGD with momentum, BPTT. expert capacity 제약으로 GPU/TPU 분산.
- 핵심 수식:

$$
y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x), \quad G(x) = \text{softmax}(\text{KeepTopK}(H(x), k))
$$

여기서 $E_i$ 는 $i$ 번째 expert, $G(x)$ 는 sparse gating (top-$k$ 외에는 0).

## 실험 / 결과

- 1B Word LM: perplexity 28.0 (이전 SOTA 31.3) at 8.7B params.
- 100B News LM: perplexity 28.0 at 137B params — 당시 최대 LSTM.
- WMT'14 En-Fr BLEU 40.56 (이전 SOTA 39.92).
- 재현성: 공식 코드는 없으나 후속 GShard / Switch 가 표준 reference 구현.

## 한계 / 비판적 시각

- LSTM 기반 — Transformer 시대 이전. routing 의 noise gating 트릭이 현대에는 잘 안 씀.
- Expert collapse 방지 보조 loss 의 hyperparameter 가 민감 — 후속 Switch Transformer 가 단순화.
- 분산 학습의 communication 비용 큼 — all-to-all 의 효율은 GShard 가 해결.
- top-$k$ 가 hard discrete 결정 → 미분 불가능, gradient 추정 어려움 (gumbel-softmax 등 후속).

## 본 프로젝트 시사점

- **패러다임의 origin 문헌** — token 이 expert graph 의 일부 노드를 선택해 routing 하는 첫 사례. `src/graphlm/models/` 의 모든 MoE 구현이 본 논문의 기본 형태를 따름.
- **차용할 핵심 요소**:
  - **Sparse gating의 안정화 보조 loss** (load-balancing) — expert 활용 균형 강제. 본 프로젝트의 첫 MoE 구현 시 default 옵션.
  - **noisy gating** — exploration 강화. 작은 dataset 의 toy 실험에서 유용 가능.
- **채택하지 않을 부분**: LSTM backbone — Transformer 로 대체. 분산 expert sharding 도 단일 GPU 검증 단계에선 생략.
- **후속 실험 가설**: $N$ (expert 수) vs $k$ (top-$k$) 의 trade-off — 작은 모델에서 $k=1$ (Switch) vs $k=2$ (original) 의 성능 / 안정성 차이 측정.

## 참고 / 인용

- 공식 코드: 없음 (PyTorch 비공식 구현 다수)
- 관련 논문: [Switch Transformer](2022-switch-transformer-fedus.md) (단순화), [GShard](2021-gshard-lepikhin.md) (scaling), [Mixtral](2024-mixtral-jiang.md) (LLM)
- 본 프로젝트 내 인용 위치: 추후 toy MoE 노트북에서
