---
title: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
authors: "Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P. C., & Santoro, A."
year: 2024
venue: "arXiv preprint (DeepMind)"
url: "https://arxiv.org/abs/2404.02258"
arxiv_id: "2404.02258"
code_url: "https://github.com/lucidrains/mixture-of-depths-pytorch"
tags: ["computation-graph", "mod", "mixture-of-depths", "dynamic-depth", "token-routing", "sparse"]
status: "draft"
cited_in: []
---

# Mixture of Depths — Per-token Layer Skip

## TL;DR (3줄)

- 각 Transformer block 마다 **expert-choice 방식의 router** 가 \"이 step 에 어떤 token 을 통과시킬지\" 결정 — 나머지 token 은 residual 로 skip.
- 학습 / 추론 시 token 별로 다른 깊이의 computation path → token-level compute budget 동적 할당.
- **MoE 가 \"어떤 expert 로 갈지\" 라면, MoD 는 \"이 layer 를 통과할지 말지\"** — sparse routing 의 깊이축 버전. 본 프로젝트 패러다임의 layer-routing 갈래 대표.

## 핵심 기여

- **Expert-choice routing for capacity control**: 각 block 의 router 가 top-$k$ token (capacity $\le N/k$) 만 선택 → load balance 가 자동 보장 (token-choice MoE 의 collapse 문제 회피).
- **Sparse compute at fixed sequence length**: token 수 $N$ 중 $\lfloor N/k \rfloor$ 만 layer 통과 → 같은 token budget 에서 FLOPs 절감, 또는 같은 FLOPs 에서 더 깊은 모델 학습.
- **Combined MoD-MoE**: 깊이 routing (MoD) 과 expert routing (MoE) 를 동시 적용 가능 — \"MoDE\" 변형.
- **Throughput improvement**: 50% sparsity 시 동일 quality 에 ~50% step time 단축.

## 방법 요약

- 데이터: C4 / SlimPajama (사전학습), HellaSwag / PIQA / ARC 등 다운스트림.
- 모델: standard decoder Transformer + 각 block 앞에 router (linear layer). $k$ 는 block 마다 (또는 동일).
- 학습: standard cross-entropy. router 는 expert-choice → load balance 보조 loss 불필요.
- 핵심 흐름:

각 block 의 router weight $r(x_t) = x_t W_r$. top-$k$ token 선택 후:

$$
y_t = \begin{cases} \text{TransformerBlock}(x_t) + x_t & \text{if } t \in \text{top-}k \\ x_t & \text{otherwise} \end{cases}
$$

## 실험 / 결과

- 360M~1B 파라미터 decoder, C4 pretraining.
- 동일 quality 에서 **50% token 만 통과** 해도 perplexity 손실 거의 없음 → step time ~50% 단축.
- 동일 FLOP budget 에서 더 깊은 (2x layer) 모델 가능 → quality 향상.
- Token-choice (MoE-style) routing 대비 expert-choice 의 안정성 큰 폭 우위.
- 재현성: 공식 코드는 없으나 lucidrains 등 PyTorch 커뮤니티 구현 활성.

## 한계 / 비판적 시각

- **Expert-choice 의 causal 한계** — autoregressive 추론 시 \"이 token 이 top-$k$ 인지\" 를 미리 알 수 없음 (모든 token 의 router score 가 필요) → causal mask 와의 호환에 트릭 필요 (논문은 학습 시 non-causal router + 추론 시 별도 처리).
- 본 논문이 비교적 신작 (2024 preprint) — 후속 검증 / 재현 사례 축적 중.
- $k$ 의 schedule (block 마다 capacity) 가 hyperparameter — 자동 선택 미해결.
- MoE 와 직교적이지만, 동시 적용 시 학습 동역학의 상호작용은 추가 분석 필요.

## 본 프로젝트 시사점

- **Layer-routing 갈래의 대표** — \"token 이 layer graph 위에서 어떤 노드를 통과할지\" 학습. Universal Transformer 의 token-level halting 의 sparse / 효율 버전.
- **차용할 아이디어**:
  - **Expert-choice routing** — token-choice (Switch) 의 load balance 문제 회피. 본 프로젝트의 MoE 구현에서도 expert-choice 변형 고려 가치.
  - **Layer 별 router head** — 작은 linear layer 만 추가하면 되는 가벼운 trick. 본 프로젝트의 첫 동적 depth 실험 reference.
- **채택하지 않을 부분**: causal autoregressive 추론의 추가 트릭 — 본 프로젝트가 encoder-only setup 으로 시작한다면 우회 가능.
- **후속 실험 가설**:
  - 작은 dataset 에서 MoD 의 50% sparsity 가 정말로 quality 유지 가능한지 (논문은 C4 사전학습 가정).
  - Switch (expert routing) + MoD (layer routing) 의 결합이 단순 추가형인지 상호작용 효과 있는지 ablation.

## 참고 / 인용

- 공식 코드: 비공식 PyTorch <https://github.com/lucidrains/mixture-of-depths-pytorch>
- 관련 논문: [Universal Transformer](2019-universal-transformer-dehghani.md) (token-level halting 의 dense 버전), [Switch Transformer](2022-switch-transformer-fedus.md) (expert routing 갈래), [Sparsely-Gated MoE](2017-moe-shazeer.md)
- 본 프로젝트 내 인용 위치: 추후 layer-routing 실험 노트북에서
