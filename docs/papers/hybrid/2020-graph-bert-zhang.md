---
title: "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
authors: "Zhang, J., Zhang, H., Xia, C., & Sun, L."
year: 2020
venue: "arXiv preprint"
url: "https://arxiv.org/abs/2001.05140"
arxiv_id: "2001.05140"
code_url: "https://github.com/jwzhanggy/Graph-Bert"
tags: ["hybrid", "graph-bert", "transformer", "subgraph", "pretraining", "no-message-passing"]
status: "draft"
cited_in: []
---

# Graph-BERT — Attention-Only Graph Representation

## TL;DR (3줄)

- Message passing 을 완전히 버리고, 각 노드 주변의 **샘플링된 sub-graph 를 token 시퀀스로 변환** 해 표준 BERT 로 학습.
- Node attribute reconstruction + structure recovery 두 사전학습 과제로 transferable representation 확보.
- "GNN ≠ message passing 만이 답이 아니다" 의 초기 증명 — over-smoothing / suspended animation 문제를 우회.

## 핵심 기여

- **Linkless subgraph batch**: 각 노드를 중심으로 **intimacy ranking** (PageRank 기반) 으로 top-$k$ 노드를 뽑아 sequence 화 → 거대 graph 도 fixed-size batch.
- **4종 positional embedding** 융합: Weisfeiler-Lehman absolute / intimacy-based relative / hop-based / raw node feature.
- **Self-supervised pre-train + fine-tune**: BERT 패턴 그대로 graph 도메인에 — node attribute reconstruction + structure recovery.

## 방법 요약

- 데이터: Cora, Citeseer, Pubmed (사전학습 + 다운스트림).
- 모델: 12-layer Transformer encoder. 각 입력은 target 노드 + intimacy top-$k$ 이웃의 token 시퀀스.
- 학습: 2-stage — (1) self-supervised pre-train, (2) supervised fine-tune (node classification / graph clustering).
- 핵심 수식: 표준 multi-head self-attention + positional embedding 합산.

## 실험 / 결과

- Cora 84.3% / Citeseer 71.2% / Pubmed 79.3% — GCN/GAT 동급 또는 약간 우위.
- Layer 50 이상에서도 성능 유지 (over-smoothing 회피 가설 검증).
- 재현성: 공식 PyTorch 공개. small graph 위주 검증 — 큰 graph 의 intimacy 전처리 비용 미보고.

## 한계 / 비판적 시각

- Intimacy ranking 의 PageRank 전처리가 dynamic graph 에 부적합.
- top-$k$ 가 작으면 long-range 정보 손실, 크면 quadratic attention 부담.
- "Only attention is needed" 라는 강한 주장 대비 성능 마진이 좁음 (이후 Graphormer 가 더 강력 증명).
- 사전학습 task 가 graph 자체 정보만 사용 → text-attributed graph 의 풍부한 텍스트 신호 미활용 (GraphFormers / GLEM 가 보완).

## 본 프로젝트 시사점

- **Graph 를 sequence 로 보는 관점**: 본 프로젝트의 LM 코어와 가장 잘 맞는 접근 — graph 를 token 시퀀스로 변환하면 기존 transformer infra 재사용 가능.
- **사전학습 패턴 reference**: BERT 식 MLM / structure recovery 의 graph 도메인 적용 예시 → 본 프로젝트의 사전학습 노트북 (`notebooks/01-experiments/`) 설계 참고.
- **차용할 아이디어**: intimacy-based subgraph sampling 은 메모리 효율적인 batch 구성에 유효. WL/hop/intimacy positional encoding 의 4-way fusion 도 참고.
- **채택하지 않을 부분**: message passing 완전 배제는 과한 선언 — 본 프로젝트는 message passing + attention 의 hybrid 채택 가능성 큼.
- **후속 실험 가설**: Graph-BERT 의 subgraph token + GraphFormers 의 GNN-nested 결합으로 작은 dataset 에서 어느 쪽 inductive bias 가 우월한지 ablation.

## 참고 / 인용

- 공식 코드: <https://github.com/jwzhanggy/Graph-Bert> (PyTorch)
- 관련 논문: [Graphormer](../graph/2021-graphormer-ying.md) (더 강력한 Graph Transformer), [GraphFormers](2021-graphformers-yang.md) (GNN + Transformer nested), [GLEM](2023-glem-zhao.md) (text-attributed graph 의 LM 활용)
- 본 프로젝트 내 인용 위치: 추후 graph-as-sequence 사전학습 노트북에서
