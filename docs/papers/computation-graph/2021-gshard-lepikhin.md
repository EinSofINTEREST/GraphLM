---
title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
authors: "Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z."
year: 2021
venue: "ICLR 2021"
url: "https://arxiv.org/abs/2006.16668"
arxiv_id: "2006.16668"
code_url: "https://github.com/google-research/google-research/tree/master/scaling_mlperf"
tags: ["computation-graph", "moe", "gshard", "scaling", "sharding", "top-2-gating", "translation"]
status: "draft"
cited_in: []
---

# GShard — MoE Scaling 의 Production Recipe

## TL;DR (3줄)

- Sparsely-Gated MoE 의 production-grade 확장 — **top-2 gating + expert capacity + random dispatch + auxiliary load balance** 의 4종 트릭으로 600B 파라미터 NMT 모델 안정 학습.
- **XLA 의 자동 sharding annotation** 으로 TPU pod 전체에 expert 를 분산 — communication 비용 최적화의 표준 reference.
- Google 의 Multilingual NMT (M4) 모델 — 100 개 언어쌍을 단일 모델로 처리하면서 dense baseline 대비 BLEU 큰 폭 개선.

## 핵심 기여

- **Top-2 gating**: top-1 의 빈약함과 top-$k$ ($k>2$) 의 비용 사이의 sweet spot. 첫 expert 는 확정, 두 번째는 확률적 (under threshold 면 drop).
- **Expert capacity 제약**: 각 expert 가 처리할 수 있는 token 수 상한 ($\text{capacity} = \frac{\text{tokens}}{\text{experts}} \cdot \text{factor}$). 초과 시 token drop → 안정성 확보.
- **Auxiliary load-balancing loss** (개선): expert 별 routing 확률 평균과 실제 fraction 의 곱을 minimize → 균형 강제.
- **Random dispatch**: 두 번째 expert 의 routing 에 noise 주입 → exploration.
- **XLA SPMD sharding**: \"이 차원을 device 축으로 split\" annotation 만으로 분산 학습 자동화.

## 방법 요약

- 데이터: Multilingual NMT (100개 언어쌍, M4 dataset, 25B sentence pairs).
- 모델: Transformer encoder/decoder + every other FFN 을 MoE layer 로 교체. expert 수 $N=128 \sim 2048$. 총 600B parameters.
- 학습: Adafactor, label smoothing. TPU v3 pod (1024~2048 cores). expert 는 각 core 에 분산.
- 핵심 흐름:

$$
\text{gate}(x) = \text{softmax}(xW_g), \quad \text{top-2 indices} = \arg\text{top2}(\text{gate}(x))
$$

각 token 은 capacity 한도 내에서 top-2 expert 로 dispatch, weighted sum 으로 출력.

## 실험 / 결과

- M4 평균 BLEU 44.3 (dense baseline 36.9, +7.4 BLEU).
- 학습 시간: 600B MoE 가 96B dense 대비 4배 효율 (BLEU 당 wall-clock).
- 재현성: 공식 구현은 Google 내부 (JAX). 후속 Switch / Mixtral 가 Pre-LayerNorm + top-1 으로 단순화 reference.

## 한계 / 비판적 시각

- TPU pod 가정 — GPU 환경에서는 communication 비용이 다름 (Switch Transformer 가 GPU 적합화).
- Token dropping (capacity 초과) 이 quality degradation 원인 — 후속 No-Token-Left-Behind (NLLB) 가 보완.
- 600B 가 학습 안정성에 도달하기까지 hyperparameter 튜닝 부담 큼.
- Routing 결정이 hard discrete → gradient 가 gate 의 soft prob 으로만 흐름 (REINFORCE 류 아님).

## 본 프로젝트 시사점

- **MoE 의 production-grade 표준** — 본 프로젝트가 MoE 모델을 학습할 때 capacity / load balance / top-2 의 default 가 본 논문에서 옴.
- **차용할 아이디어**:
  - **Top-2 gating** — top-1 (Switch) 보다 안정적이고 표현력 높음. 첫 MoE 구현 시 권장.
  - **Capacity factor** — token drop 률을 trade-off 로 조절. 본 프로젝트 PyTorch 구현 시 hyperparameter 노출.
  - **Load balance loss formulation** — 정확한 수식 그대로 채용.
- **채택하지 않을 부분**: XLA SPMD 기반 sharding — 본 프로젝트는 단일 GPU 또는 작은 cluster 우선. PyTorch FSDP + custom expert parallelism 으로 대체.
- **후속 실험 가설**: capacity factor 의 변화 (1.0 vs 1.5 vs 2.0) 가 작은 dataset 에서 token drop률 vs 정확도에 미치는 영향. Switch (top-1) 대비 GShard (top-2) 의 effective sparsity 차이.

## 참고 / 인용

- 공식 코드: 부분 (Google JAX 내부)
- 관련 논문: [Sparsely-Gated MoE](2017-moe-shazeer.md) (직계 조상), [Switch Transformer](2022-switch-transformer-fedus.md) (top-1 단순화), [Mixtral](2024-mixtral-jiang.md) (현대 적용)
- 본 프로젝트 내 인용 위치: 추후 MoE 학습 hyperparameter ablation 노트북에서
