---
title: "GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph"
authors: "Yang, J., Liu, Z., Xiao, S., Li, C., Lian, D., Agrawal, S., Singh, A., Sun, G., & Xie, X."
year: 2021
venue: "NeurIPS 2021"
url: "https://arxiv.org/abs/2105.02605"
arxiv_id: "2105.02605"
code_url: "https://github.com/microsoft/GraphFormers"
tags: ["hybrid", "graphformers", "text-attributed-graph", "transformer", "gnn-nested", "joint"]
status: "draft"
cited_in: []
---

# GraphFormers — GNN-nested Transformers

## TL;DR (3줄)

- 문서(텍스트 노드) 와 graph 구조를 함께 학습 — Transformer layer **사이마다 GNN aggregation 을 끼워 넣어** text 토큰 표현이 이웃 문서 정보로 반복 업데이트.
- "텍스트 인코딩 → 그 출력으로 GNN" 의 단순 cascade 대비, 두 단계가 매 layer 마다 상호작용 → text + graph 가 깊이 융합.
- 대규모 link prediction (Wikidata5M, OAG) 에서 BERT+GNN cascade baseline 대비 큰 폭 개선.

## 핵심 기여

- **GNN-nested Transformer**: 표준 Transformer block 마다 `[CLS]` 토큰들이 graph aggregation 으로 이웃 문서의 `[CLS]` 를 흡수 → 텍스트 인코딩 도중 graph context 가 점진 주입.
- **Two-stage pretraining (link prediction)**: 큰 corpus 의 link prediction 으로 사전학습 → 다운스트림 fine-tune.
- **Asymmetric design**: query 문서는 full encoding, 이웃 문서는 partial encoding (속도 ↑).

## 방법 요약

- 데이터: Wikidata5M (Wikipedia entity graph), OAG (Open Academic Graph) — 문서가 노드, 인용/관계가 edge.
- 모델: BERT-base 위에 N=2~3 의 nested GNN aggregation (mean / attention).
- 학습: link prediction loss (positive vs sampled negative neighbors), AdamW.
- 핵심 흐름:

$$
\text{layer-}l: \quad h_v^{(l)} \leftarrow \text{Transformer}\!\left(h_v^{(l-1)}\right);\quad
[\text{CLS}]_v^{(l)} \leftarrow \text{Agg}\!\left(\{[\text{CLS}]_u^{(l)} : u \in \mathcal{N}(v)\}\right)
$$

## 실험 / 결과

- Wikidata5M link prediction MRR 0.395 vs BERT-only 0.328, GNN-only 0.250.
- OAG 도 유사한 폭의 개선.
- 추론 비용은 cascade 대비 약간 증가 (~1.5x), 학습 비용은 비슷.
- 재현성: 공식 PyTorch (Microsoft) 공개.

## 한계 / 비판적 시각

- Negative sampling 의 strategy 에 민감 (link prediction 의 표준 문제).
- 이웃 수가 크면 layer 마다 aggregation 비용 누적 — fan-out 제한 필요.
- 텍스트 길이가 짧은 노드 (예: 짧은 캡션) 에는 nested aggregation 의 효과 미보고.
- 후속 GLEM 가 EM 방식으로 더 단순 + 강력하게 GNN/LM 협업을 제안.

## 본 프로젝트 시사점

- **본 프로젝트의 핵심 hybrid 패턴 후보**: text-attributed graph 가 GraphLM 의 주력 setting 이라면 GraphFormers 는 가장 가까운 reference architecture.
- **모듈 재사용**: `src/graphlm/models/` 에 `GnnNestedTransformer` 를 정의할 때, 본 논문의 layer-loop 가 가장 깔끔한 추상화. PyTorch 의 `nn.TransformerEncoderLayer` 를 sub-classing + post-hook 으로 구현 가능.
- **차용할 아이디어**: query / neighbor 비대칭 인코딩 (full vs partial) 은 큰 graph 에서 inference 비용을 현실화하는 핵심 트릭.
- **채택하지 않을 부분**: link prediction 만의 pretrain 은 본 프로젝트 목적이 다를 시 MLM + structure 등으로 대체.
- **후속 실험 가설**: nested aggregation 의 depth (몇 번째 layer 부터 graph 주입) 와 다운스트림 성능의 관계 ablation — Graph-BERT/Graphormer 의 attention bias 와 직접 비교.

## 참고 / 인용

- 공식 코드: <https://github.com/microsoft/GraphFormers> (PyTorch)
- 관련 논문: [[2020-graph-bert-zhang]] (graph-only Transformer), [[2021-graphormer-ying]] (structural bias 접근), [[2023-glem-zhao]] (EM 기반 단순 co-training)
- 본 프로젝트 내 인용 위치: 추후 text-attributed graph hybrid 실험 노트북에서
