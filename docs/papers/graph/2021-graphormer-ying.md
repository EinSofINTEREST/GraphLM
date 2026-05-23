---
title: "Do Transformers Really Perform Bad for Graph Representation?"
authors: "Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., & Liu, T.-Y."
year: 2021
venue: "NeurIPS 2021"
url: "https://arxiv.org/abs/2106.05234"
arxiv_id: "2106.05234"
code_url: "https://github.com/microsoft/Graphormer"
tags: ["graph", "gnn", "graphormer", "transformer", "attention-bias", "molecular"]
status: "draft"
cited_in: []
---

# Graphormer — Transformer for Graph Representation

## TL;DR (3줄)

- 표준 Transformer encoder 에 **3종 structural bias** (centrality / spatial / edge) 만 더해 graph 전용 모델로 변환.
- OGB-LSC PCQM4M-LSC quantum chemistry challenge 1위 — 분자 그래프에서 message-passing GNN 의 우위 가설을 뒤집음.
- "Transformer 가 graph 에서도 가능하다" 의 결정적 선례 → 이후 Graph Transformer 연구 폭증의 출발점.

## 핵심 기여

- **Centrality encoding**: 각 노드의 in/out degree 를 learnable embedding 으로 입력에 더함 → 노드 중요도 신호.
- **Spatial encoding**: 노드 쌍 사이 최단경로 길이 $\phi(v_i, v_j)$ 에 따른 learnable bias 를 attention score 에 더함 → 그래프 거리 인식.
- **Edge encoding**: 최단경로상의 edge feature 평균을 attention bias 에 추가 → edge attribute 가 풍부한 분자 graph 에 효과적.

## 방법 요약

- 데이터: OGB-LSC PCQM4M-LSC (HOMO-LUMO gap 회귀, 380만 분자), ZINC, OGB Mol-HIV/PCBA
- 모델: 12-layer Transformer encoder + 위 3 bias.
- 학습: AdamW, learning rate warmup, MAE (회귀) / cross-entropy (분류).
- 핵심 수식:

$$
A_{ij} = \frac{(h_i W_Q)(h_j W_K)^\top}{\sqrt{d}} + b^{\phi(v_i,v_j)} + c_{ij}
$$

여기서 $b^{\phi}$ 는 spatial bias, $c_{ij}$ 는 shortest-path edge encoding.

## 실험 / 결과

- PCQM4M-LSC validate MAE **0.1234** → 대회 1위 (GIN/MPNN baseline 대비 큰 폭).
- ZINC test MAE 0.122 (small dataset).
- OGB Mol-HIV ROC-AUC 80.5%.
- 재현성: 공식 PyTorch (Microsoft) 공개. 분자 도메인은 안정적이나 일반 graph 로 확장 시 spatial bias 의 비용 큼.

## 한계 / 비판적 시각

- **Pairwise spatial bias** 가 $O(N^2)$ 메모리 → 큰 graph 확장 어려움 (NodeFormer, SGFormer 등이 보완).
- Shortest-path 계산이 전처리 비용 — dynamic graph 어려움.
- 분자/소규모 graph 에서 강력, 대규모 sparse graph (소셜 네트워크, 추천) 에선 추가 트릭 필요.
- Position 정보가 spatial encoding 에 의존 → 비연결 graph 처리 명시 필요.

## 본 프로젝트 시사점

- **Graph 와 Transformer 의 가교**: 본 프로젝트가 "Graph + LM" 을 다룬다면 Graphormer 의 3 bias 패턴은 LM 의 attention 모듈에 structural prior 를 주입하는 reference design.
- **분자/소규모 graph 우선**: 본 프로젝트의 초기 실험이 소~중 규모 dataset 위주라면 Graphormer 는 직접 채택 후보 — GCN/GAT baseline 위 attention-bias upgrade.
- **차용할 아이디어**: spatial bias 의 learnable scalar 형태는 본 프로젝트의 hybrid attention 에 plug-in 가능. `attention_score += spatial_bias_table[shortest_path_len]`.
- **채택하지 않을 부분**: 대규모 graph 시 $O(N^2)$ — 본 프로젝트가 거대 graph 다룰 가능성 있다면 NodeFormer 류 linear attention 으로 대체.
- **후속 실험 가설**: LM 의 token-level attention + Graphormer 의 node-level spatial bias 를 두 stream 으로 합치면 small text-attributed graph 에서 GLEM 류 co-training 보다 단순/효과적인지.

## 참고 / 인용

- 공식 코드: <https://github.com/microsoft/Graphormer>
- 관련 논문: [[2018-gat-velickovic]] (attention 도입 선례), [[2020-graph-bert-zhang]] (다른 Graph Transformer 방향), [[2021-graphformers-yang]] (Graph + LM Transformer 결합)
- 본 프로젝트 내 인용 위치: 추후 attention-bias 실험 노트북에서
