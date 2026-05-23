---
title: "Learning on Large-scale Text-attributed Graphs via Variational Inference"
authors: "Zhao, J., Qu, M., Li, C., Yan, H., Liu, Q., Li, R., Xie, X., & Tang, J."
year: 2023
venue: "ICLR 2023"
url: "https://arxiv.org/abs/2210.14709"
arxiv_id: "2210.14709"
code_url: "https://github.com/AndyJZhao/GLEM"
tags: ["hybrid", "glem", "text-attributed-graph", "lm-gnn", "em", "co-training", "variational"]
status: "draft"
cited_in: []
---

# GLEM — Graph and Language Learning via EM

## TL;DR (3줄)

- 대규모 text-attributed graph 에서 LM 과 GNN 을 **EM (Expectation-Maximization) 으로 번갈아 학습** — 두 모델이 서로의 출력을 pseudo-label 로 활용.
- LM 과 GNN 을 end-to-end joint 로 묶지 않아 **각자의 batching 패러다임 보존** → 큰 graph 에서 메모리 효율적이며 분산 학습 친화적.
- OGB 의 ogbn-arxiv/products/papers100M 에서 큰 폭의 SOTA — text + structure 모두 활용 시 강력함을 단순한 framework 로 입증.

## 핵심 기여

- **Variational view of text-attributed graph learning**: 노드 라벨을 latent 로 두고 LM 의 텍스트 우도 + GNN 의 구조 우도를 ELBO 로 결합 → 이론적으로 깔끔한 EM 반복.
- **Decoupled training**: LM step 은 텍스트만, GNN step 은 LM embedding + structure 만 → 두 모델이 동시에 GPU 에 올라가지 않음 (GraphFormers/엔드투엔드 nested 의 메모리 한계 해소).
- **Pseudo-label injection**: 각 step 에서 다른 모델의 confident prediction 을 unlabeled 노드의 학습 신호로 — semi-supervised 효과까지 확보.

## 방법 요약

- 데이터: OGB ogbn-arxiv (17만 노드), ogbn-products (240만), ogbn-papers100M (1억).
- 모델: LM = DeBERTa-base / GNN = GraphSAGE 또는 RevGAT.
- 학습 루프: E-step (LM 가 텍스트 → soft label) → M-step (GNN 가 graph + 그 soft label 로 갱신) → 다시 LM 갱신. K=2~3 iterations.
- 핵심 수식 (ELBO):

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(Y_U)}[\log p_{\text{LM}}(X|Y) + \log p_{\text{GNN}}(Y|G)] - \text{KL}(q\,\|\,p)
$$

## 실험 / 결과

- ogbn-arxiv accuracy **76.97%** vs 이전 SOTA (RevGAT) 74.0% — 큰 폭.
- ogbn-products **87.36%**, ogbn-papers100M **70.86%** — 모두 leaderboard 상위.
- LM 단독 / GNN 단독 / cascade 대비 일관되게 우위.
- 재현성: 공식 PyTorch 공개. OGB 표준 split + 시드로 안정 재현.

## 한계 / 비판적 시각

- 텍스트가 풍부한 graph (academic citations) 위주 검증 — short-text / sparse-text graph 의 효과는 미보고.
- EM iteration 수 / pseudo-label confidence threshold 가 hyperparameter → tuning 부담.
- LM 과 GNN 의 표현 공간이 **직접 align 되지 않음** — 두 모델의 feature 가 서로 다른 의미 공간에 살 가능성 (다운스트림 zero-shot transfer 어려움).
- decoupling 의 대가로 두 모델 간 gradient 공유 안 됨 → 표현 융합 깊이는 GraphFormers 가 더 깊음.

## 본 프로젝트 시사점

- **본 프로젝트의 default hybrid 패턴 후보 1순위**: 단순함 + 확장성 + 강력함 → text-attributed graph 가 GraphLM 의 주력 setting 일 때 가장 안전한 출발점.
- **모듈 분리에 친화적**: 본 프로젝트의 `src/graphlm/models/{lm,gnn}/` 분리 구조에 자연스럽게 매핑 → EM loop 만 `src/graphlm/training/glem_loop.py` 형태로 추가.
- **차용할 아이디어**: pseudo-label 의 confidence-weighted KL — 작은 dataset 에서도 unlabeled 노드 활용 가능.
- **채택하지 않을 부분**: LM 과 GNN 표현 공간 misalignment — 본 프로젝트가 zero-shot / cross-task transfer 를 노린다면 GraphFormers 식 joint embedding 으로 보완 필요.
- **후속 실험 가설**: GLEM 의 EM iteration 수와 다운스트림 정확도의 trade-off (cost vs gain) 를 작은 dataset (Cora text) 에서 측정 → 본 프로젝트 default K 결정 근거.

## 참고 / 인용

- 공식 코드: <https://github.com/AndyJZhao/GLEM> (PyTorch)
- 관련 논문: [[2021-graphformers-yang]] (end-to-end nested 대비군), [[2020-graph-bert-zhang]] (graph-side Transformer), [[2024-graphgpt-tang]] (LLM 시대로의 확장)
- 본 프로젝트 내 인용 위치: 추후 EM-based hybrid training 노트북에서
