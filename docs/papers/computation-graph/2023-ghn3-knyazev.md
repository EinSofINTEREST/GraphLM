---
title: "Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models?"
authors: "Knyazev, B., Hwang, D., & Lacoste-Julien, S."
year: 2023
venue: "ICML 2023"
url: "https://arxiv.org/abs/2303.04143"
arxiv_id: "2303.04143"
code_url: "https://github.com/SamsungSAILMontreal/ghn3"
tags: ["computation-graph", "ghn", "ghn3", "hypernetwork", "architecture-as-graph", "parameter-prediction"]
status: "draft"
cited_in: []
---

# GHN-3 — Graph HyperNetwork for Neural Architecture

## TL;DR (3줄)

- 신경망 아키텍처 자체를 **DAG (directed acyclic graph)** 로 표현 — 각 layer/연산이 노드, dataflow 가 edge.
- 거대 Transformer (\"GHN-3\") 가 그 DAG 을 입력 받아 **임의 unseen architecture 의 weight 를 직접 predict** — 학습 없이 ImageNet accuracy 60% 이상 달성.
- 본 프로젝트 패러다임의 **\"NN architecture = graph\"** 측면을 가장 직접적으로 구현한 사례. 노드 = computation operation, 학습 대상 = 그 노드의 parameter.

## 핵심 기여

- **Architecture-as-DAG with rich node features**: 각 노드(연산) 마다 type, kernel size, stride, in/out channels 등의 메타데이터를 임베딩. edge 는 dataflow.
- **Transformer-based hypernetwork**: 기존 GHN-2 (GNN 기반) 를 Transformer (위치 임베딩 + self-attention) 로 대체 → 더 큰 architecture diversity 처리.
- **Massive training**: ImageNet 의 1M 개 diverse architecture 데이터셋 (DeepNets-1M) 으로 사전학습. 후속 unseen architecture 에 대해 zero-shot 가능.
- **Tied multi-task scaling**: 단일 hypernetwork 가 ResNet / ViT / DenseNet 등 모두 동일 framework 에서 처리.

## 방법 요약

- 데이터: DeepNets-1M (1M architectures), ImageNet (downstream).
- 모델:
  - Input: target architecture 의 DAG (각 노드 = layer/op)
  - GHN-3: 24-layer Transformer with 비대칭 self-attention. node 시퀀스 (topological sort) 입력.
  - Output: 각 노드의 weight tensor (shape 다양 — 임베딩으로 shape 정보 명시).
- 학습: MSE between predicted weight 와 SGD-trained weight (teacher).
- 핵심 흐름:

$$
\theta_v = f_{\text{GHN-3}}(\text{node features}_v, \text{context from neighbors})
$$

각 노드 $v$ 의 parameter $\theta_v$ 를 다른 노드의 context 를 보고 예측.

## 실험 / 결과

- ImageNet zero-shot top-1 (학습 없이 hypernetwork 가 predict 한 weight 로 평가):
  - GHN-3-XL: 60.0% (이전 GHN-2: 5%)
- Fine-tune 시작점으로 사용: GHN-3 prediction 으로 init → 빠른 수렴.
- Architecture transfer: ResNet 으로 학습 → ViT 의 weight 예측 가능 (cross-architecture generalization).
- 재현성: 공식 PyTorch (Samsung SAIL Montreal) 공개.

## 한계 / 비판적 시각

- 예측된 weight 의 accuracy 60% 는 SGD-trained 의 70~80% 에 못 미침 — fine-tune 없이는 production 수준 어려움.
- DAG 표현이 \"고정된 op 집합\" 가정 — 새 op (예: novel attention 변형) 는 처리 불가.
- 학습 데이터 (DeepNets-1M) 자체가 1M arch 의 SGD 학습 결과 — 생성 비용 막대.
- Vision (ImageNet) 위주 검증 — NLP / LM 의 hypernetwork 는 미보고.

## 본 프로젝트 시사점

- **\"NN architecture = graph\" 패러다임의 가장 명시적 사례** — GraphLM 이 \"Transformer 내부 구조를 graph 로\" 패러다임을 가질 때, GHN-3 의 DAG 표현이 그 graph 의 reference schema.
- **차용할 아이디어**:
  - **Node feature schema** (op type, dim, kernel 등) — 본 프로젝트가 Transformer architecture 를 데이터로 다룬다면 직접 적용.
  - **Topological order + Transformer** — DAG 의 sequence 화 → 표준 Transformer 처리 가능. 일반 GNN 대비 implementation 단순.
- **채택하지 않을 부분**: vision-specific search space (Conv, BatchNorm 등) — 본 프로젝트는 Transformer 내부 sub-module 위주의 좁은 schema.
- **후속 실험 가설**:
  - MoE Transformer 의 \"expert configuration\" 자체를 graph 로 보고 GHN 류 hypernetwork 로 expert weight 를 predict 하는 micro-experiment.
  - GHN-3 의 \"weight prediction + fine-tune\" 패러다임이 small dataset 에서 random init 보다 의미 있는지.

## 참고 / 인용

- 공식 코드: <https://github.com/SamsungSAILMontreal/ghn3>
- 관련 논문: [AutoFormer](2021-autoformer-chen.md) (NAS 관점), \"Parameter Prediction for Unseen Deep Architectures\" (Knyazev 2021, GHN-2)
- 본 프로젝트 내 인용 위치: 추후 architecture 자체를 학습 대상으로 보는 실험에서
