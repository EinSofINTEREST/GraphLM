---
title: "Inductive Representation Learning on Large Graphs"
authors: "Hamilton, W. L., Ying, R., & Leskovec, J."
year: 2017
venue: "NeurIPS 2017"
url: "https://arxiv.org/abs/1706.02216"
arxiv_id: "1706.02216"
code_url: "https://github.com/williamleif/GraphSAGE"
tags: ["graph", "gnn", "graphsage", "inductive", "sampling", "baseline"]
status: "draft"
cited_in: []
---

# GraphSAGE — Sample and Aggregate

## TL;DR (3줄)

- 학습 시 보지 못한 노드/그래프에도 embedding 을 생성하는 **inductive** GNN framework.
- 각 노드의 K-hop 이웃을 **샘플링** 한 뒤 학습된 aggregator (mean/LSTM/pool) 로 정보를 모음 → 대규모 그래프 / 동적 그래프 / 새 노드 추가 지원.
- 산업계 거대 그래프 (Reddit, PPI) 에서 GCN 대비 강력한 inductive 성능 + 학습/추론 모두 mini-batch 가능.

## 핵심 기여

- **Inductive learning**: 학습 후 새 노드/sub-graph 에도 즉시 embedding 생성. transductive GCN 의 근본적 한계 돌파.
- **이웃 샘플링**: layer 별 고정 크기 이웃 추출 → 메모리/시간 비용을 그래프 크기와 독립화 (full-batch 대비 수십~수백 배 효율).
- **Aggregator 추상화**: mean/pooling/LSTM 등 학습 가능한 함수로 generalize → 후속 모든 message-passing 프레임워크의 모델.

## 방법 요약

- 데이터: Citation (Cora, Citeseer), Reddit (학습 23만 / 평가 11만 노드), PPI (multi-graph)
- 모델: K-layer SAGE — 각 layer 에서 노드 $v$ 가 이웃 집합 $\mathcal{N}(v)$ 의 fixed-size 샘플을 aggregate 후 자기 표현과 결합.
- 학습: unsupervised (random-walk based loss) 또는 supervised (node classification). Adam.
- 핵심 수식:

$$
h_v^{(l)} = \sigma\!\left(W^{(l)} \cdot \text{CONCAT}\!\left(h_v^{(l-1)}, \; \text{AGG}_l\!\left(\{h_u^{(l-1)} : u \in \mathcal{S}(v)\}\right)\right)\right)
$$

여기서 $\mathcal{S}(v)$ 는 $\mathcal{N}(v)$ 에서 고정 크기로 샘플된 이웃.

## 실험 / 결과

- Reddit: micro-F1 95.0 (GCN 기반 baseline 대비 +3~10%p, GCN 은 transductive 라 비교군 한정).
- PPI multi-graph inductive: micro-F1 60.0 (이전 SOTA 의 ~2배).
- 추론 latency: full GCN convolution 대비 batch 추론 가능 → 실시간 추천 시스템 후보.
- 재현성: 공식 코드 + 후속 PyG / DGL 구현이 표준 baseline 으로 채택.

## 한계 / 비판적 시각

- 이웃 샘플링이 random 이라 high-degree 노드에서 분산 큼 (FastGCN, GraphSAINT 가 개선).
- aggregator 선택 (mean vs LSTM) 은 task dependent 라 hyperparameter 튜닝 부담.
- 여전히 attention 없음 → 이웃의 중요도 차등은 GAT 가 도입.
- 큰 graph 의 long-range 의존성은 K 가 작으면 포착 어려움.

## 본 프로젝트 시사점

- **두 번째 baseline**: GCN 이 transductive 한계를 보이는 시점에 GraphSAGE 가 자연스러운 다음 단계. inductive eval 셋업 필수일 때 채택.
- **mini-batch 학습 패턴**: 본 프로젝트의 `src/graphlm/data/` collate / sampler 설계는 GraphSAGE 의 neighbor sampling 을 reference 로 — `torch_geometric.loader.NeighborLoader` 활용 가능성 검토.
- **차용할 아이디어**: aggregator 추상화 — `aggregate_neighbors(method=...)` 의 dispatch 구조에 직접 반영. unsupervised 학습 옵션 (random walk loss) 은 text-attributed graph 의 사전학습에 응용 후보.
- **채택하지 않을 부분**: LSTM aggregator 의 permutation 비-equivariance 는 graph 특성과 충돌 — mean/pool 중심으로.
- **후속 실험 가설**: GraphSAGE 의 inductive embedding 위에 LM 임베딩을 concat 한 simple hybrid 가 GLEM 류 복잡한 co-training 없이도 의미 있는 성능 보이는지.

## 참고 / 인용

- 공식 코드: <https://github.com/williamleif/GraphSAGE> (TensorFlow), PyG / DGL 구현 표준.
- 관련 논문: [[2017-gcn-kipf]] (transductive baseline), [[2018-gat-velickovic]] (attention 도입)
- 본 프로젝트 내 인용 위치: 추후 inductive baseline 노트북에서
