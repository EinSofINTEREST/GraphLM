---
title: "Graph Attention Networks"
authors: "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y."
year: 2018
venue: "ICLR 2018"
url: "https://arxiv.org/abs/1710.10903"
arxiv_id: "1710.10903"
code_url: "https://github.com/PetarV-/GAT"
tags: ["graph", "gnn", "gat", "attention", "inductive", "baseline"]
status: "draft"
cited_in: []
---

# GAT — Graph Attention Networks

## TL;DR (3줄)

- 각 노드가 이웃마다 **학습된 attention 가중치**로 message 를 weighted-sum 하는 GNN.
- GCN 의 fixed normalization 한계 (이웃 중요도 균등) 를 풀고, GraphSAGE 와 달리 random sampling 없이도 inductive.
- multi-head attention 으로 Cora/Citeseer/Pubmed/PPI 전 종목 SOTA — Transformer 식 attention 의 graph 도입 효시.

## 핵심 기여

- **Self-attention on graphs**: 노드 $i$ 와 이웃 $j$ 의 representation 으로 attention coefficient $\alpha_{ij}$ 학습 → 이웃별 weight 가 데이터 dependent.
- **Inductive + masked attention**: 이웃집합만으로 계산 → 새 그래프에도 즉시 적용 (GraphSAGE 의 이웃 샘플링 없이도 inductive).
- **Multi-head**: $K$ 개 독립 attention 평균/concat → 안정성과 표현력 동시 확보.

## 방법 요약

- 데이터: Cora, Citeseer, Pubmed (transductive) + PPI (inductive multi-graph)
- 모델: 2-layer GAT, multi-head (8 heads, $F'=8$ dims).
- 학습: cross-entropy, Adam, dropout (on attention weights AND inputs).
- 핵심 수식:

$$
\alpha_{ij} = \frac{\exp\!\left(\text{LeakyReLU}(a^\top [Wh_i \| Wh_j])\right)}{\sum_{k \in \mathcal{N}(i)} \exp\!\left(\text{LeakyReLU}(a^\top [Wh_i \| Wh_k])\right)}
$$

$$
h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)
$$

## 실험 / 결과

- Cora 83.0%, Citeseer 72.5%, Pubmed 79.0% — GCN 대비 +1~2%p 의 SOTA.
- PPI inductive micro-F1 **97.3** — GraphSAGE-LSTM (60.0) 대비 큰 폭 개선.
- 재현성: 공식 TF 코드 + DGL/PyG 표준 구현 존재.

## 한계 / 비판적 시각

- Attention 계산이 edge 수에 비례 → 큰 dense graph 에서 메모리 부담.
- **Static attention** 한계 — Brody et al. (GATv2, 2022) 가 지적: $a$ 와 $W$ 의 합성이 query 와 key 의 상호작용을 제약 → GATv2 가 보완.
- Layer 수 증가 시 over-smoothing 여전.
- positional/구조 정보를 명시적으로 주지 않음 → Graphormer 가 보완.

## 본 프로젝트 시사점

- **세 번째 baseline**: GCN / GraphSAGE 와 함께 표준 baseline 트리오. text-attributed graph 에서 단어/문서 노드 중요도 차등이 의미 있으므로 hybrid 실험에 자연스러운 후보.
- **Attention 의 통일 인터페이스**: 본 프로젝트의 `src/graphlm/models/` 에서 attention layer 를 정의할 때 graph 와 transformer 가 동일 abstraction 으로 표현 가능 → GAT 가 그 가교.
- **차용할 아이디어**: multi-head + edge-level attention coefficient — Graphormer 의 attention bias 와 함께 본 프로젝트 attention 모듈의 design pattern reference.
- **채택하지 않을 부분**: 원논문의 static attention (GATv2 가 우월) — 직접 구현 시 GATv2 형태를 default.
- **후속 실험 가설**: GAT 의 $\alpha_{ij}$ 와 BERT-like LM 의 token attention 을 align 하면 attention map 가 의미적으로 일관되는지 (interpretable hybrid).

## 참고 / 인용

- 공식 코드: <https://github.com/PetarV-/GAT> (TensorFlow), PyG `GATConv` / `GATv2Conv` 표준.
- 관련 논문: [GCN](2017-gcn-kipf.md) (fixed-weight predecessor), [GraphSAGE](2017-graphsage-hamilton.md) (다른 inductive 접근), [Graphormer](2021-graphormer-ying.md) (attention 확장)
- 본 프로젝트 내 인용 위치: 추후 attention-based baseline 노트북에서
