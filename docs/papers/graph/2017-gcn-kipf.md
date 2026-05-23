---
title: "Semi-Supervised Classification with Graph Convolutional Networks"
authors: "Kipf, T. N., & Welling, M."
year: 2017
venue: "ICLR 2017"
url: "https://arxiv.org/abs/1609.02907"
arxiv_id: "1609.02907"
code_url: "https://github.com/tkipf/gcn"
tags: ["graph", "gnn", "gcn", "semi-supervised", "spectral", "baseline"]
status: "draft"
cited_in: []
---

# GCN — Graph Convolutional Networks

## TL;DR (3줄)

- Spectral graph convolution 을 1차 Chebyshev 근사로 단순화해 임의 그래프에 적용 가능한 layer 를 제안.
- 인접행렬에 self-loop 와 symmetric normalization 만 더하면 단 2-layer 만으로도 Cora/Citeseer/Pubmed 의 노드 분류에서 큰 성과.
- "GNN 의 표준 baseline" 으로 자리잡아 이후 거의 모든 graph representation 논문의 비교군.

## 핵심 기여

- Localized spectral filter 의 1차 근사 → 행렬 곱 한 번 + 비선형 활성으로 한 layer 를 표현
- Renormalization trick: $\tilde{A} = A + I$, $\tilde{D}$ 는 그 degree 행렬 → $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$
- Semi-supervised 노드 분류에서 매우 적은 라벨로도 강력한 성능 (Cora: 라벨 20개/클래스로 81.5% accuracy)

## 방법 요약

- 데이터: Cora, Citeseer, Pubmed (citation graph)
- 모델: 2-layer GCN — $Z = \text{softmax}(\hat{A}\,\text{ReLU}(\hat{A}XW^{(0)})W^{(1)})$
- 학습: cross-entropy on labeled nodes, Adam, dropout, weight decay
- 핵심 수식:

$$
H^{(l+1)} = \sigma\!\left(\hat{A} H^{(l)} W^{(l)}\right)
$$

여기서 $\hat{A}$ 는 self-loop 추가 + symmetric normalize 된 인접행렬, $H^{(l)}$ 은 $l$ 번째 layer 의 노드 표현.

## 실험 / 결과

- 벤치마크: Cora 81.5% / Citeseer 70.3% / Pubmed 79.0% (모두 SOTA at the time)
- Spectral CNN, DeepWalk, ICA 등 이전 방법 대비 +5~10%p
- 재현성: 공식 TensorFlow 구현 공개, 시드 고정 시 안정. 후속 작업이 다수 재구현.

## 한계 / 비판적 시각

- **Transductive** — 학습 시 graph 전체를 봐야 함. 새로운 노드/그래프에 inductive 적용 어려움 (GraphSAGE 가 보완)
- Receptive field 가 layer 수에 비례 → 4 layer 이상에서 over-smoothing 발생
- $\hat{A}$ 가 fixed → edge 가 변하는 dynamic graph 에 한계
- 큰 그래프에서 full-batch 학습 메모리 부담

## 본 프로젝트 시사점

- **첫 baseline**: GraphLM 의 `src/graphlm/models/` 에 가장 먼저 들어갈 모델. 모든 후속 비교의 기준점.
- **데이터 흐름 검증**: $X$, $\hat{A}$, $W^{(l)}$ 의 shape contract 가 단순 → 본 프로젝트의 graph utility (`src/graphlm/graph/`) API 가 GCN forward 를 자연스럽게 표현하는지로 설계 점검 가능.
- **차용할 아이디어**: renormalization trick 은 GraphLM 의 모든 spectral-style layer 에 default 옵션. `aggregate_neighbors(method="gcn")` 의 reference 구현.
- **채택하지 않을 부분**: transductive 가정 — 본 프로젝트는 inductive setting 도 다룰 가능성이 커서 GCN 만으론 부족 (GraphSAGE 와 함께 가야 함).
- **후속 실험 가설**: 2-layer GCN + 단순 LM embedding concatenation 으로도 small text-attributed graph 에서 hybrid baseline 작동 여부.

## 참고 / 인용

- 공식 코드: <https://github.com/tkipf/gcn> (TensorFlow), PyTorch 포트 다수
- 관련 논문: [[2017-graphsage-hamilton]] (inductive 확장), [[2018-gat-velickovic]] (attention 도입)
- 본 프로젝트 내 인용 위치: 추후 `notebooks/10-experiments/` 의 baseline 노트북에서
