---
title: "GraphGPT: Graph Instruction Tuning for Large Language Models"
authors: "Tang, J., Yang, Y., Wei, W., Shi, L., Su, L., Cheng, S., Yin, D., & Huang, C."
year: 2024
venue: "SIGIR 2024"
url: "https://arxiv.org/abs/2310.13023"
arxiv_id: "2310.13023"
code_url: "https://github.com/HKUDS/GraphGPT"
tags: ["hybrid", "graphgpt", "llm", "instruction-tuning", "graph-token", "zero-shot"]
status: "draft"
cited_in: []
---

# GraphGPT — Graph Instruction Tuning for LLMs

## TL;DR (3줄)

- 사전학습된 LLM (Vicuna) 에 **graph encoder 출력을 "graph token" 으로 주입** 하고 instruction tuning 으로 graph 이해 능력을 학습.
- 2-stage tuning — (1) self-supervised graph-text alignment, (2) task-specific instruction tuning — 으로 **zero-shot 일반화** 와 **supervised 정확도** 동시 달성.
- LLM 의 추론 능력을 graph downstream task 로 transfer 한 대표 사례 — graph 가 텍스트 modality 옆에서 first-class citizen.

## 핵심 기여

- **Graph token injection**: 사전학습 graph encoder (예: GraphTransformer) 의 노드 embedding 을 lightweight projector 로 LLM token space 에 mapping — 텍스트 prompt 와 자연스럽게 concat.
- **Dual-stage instruction tuning**: (1) graph 와 그 텍스트 설명을 align (graph-matching task) → (2) downstream task instructions (node classification, link prediction) 로 fine-tune.
- **Zero-shot transfer**: 학습 시 보지 못한 graph / task 에도 일반화 — LLM 의 in-context reasoning + graph token 의 결합 효과.

## 방법 요약

- 데이터: OGB-arxiv, PubMed, Cora (학술 graph); cross-dataset / cross-task zero-shot eval.
- 모델: Vicuna-7B (LLM) + 사전학습 GraphTransformer (encoder) + projector (작은 MLP).
- 학습: 2-stage. instruction template 으로 graph task 를 자연어 prompt 로 표현 → LLM 생성을 정답과 비교.
- 핵심 흐름:

$$
\text{prompt} = \text{Instruction} \,\Vert\, \langle\text{graph\_token}\rangle \,\Vert\, \text{Query}\,;\quad
\langle\text{graph\_token}\rangle = \text{Projector}(\text{GraphEnc}(G))
$$

## 실험 / 결과

- Cora supervised accuracy 84.5%, PubMed 89.8% — pure GNN baseline 동급 또는 우위.
- Zero-shot cross-dataset (학습: arXiv → 평가: Cora/PubMed) 에서 GNN baseline 대비 +20%p 이상 — LLM 의 일반화 능력 입증.
- 재현성: 공식 PyTorch + Vicuna 가중치 공개. LLM inference 비용은 큼.

## 한계 / 비판적 시각

- **LLM inference 비용** — Vicuna-7B 라도 GNN 보다 훨씬 느림 → production 적용 한계.
- Graph encoder 와 LLM 의 alignment 가 projector 의 capacity 에 의존 → 큰 graph 의 정보를 작은 token 으로 압축하는 한계.
- 평가가 academic graph 위주 — 일반 graph (소셜, 분자, 추천) 의 효과 미확인.
- "graph 가 정말 LLM 의 추론을 받는가" vs "LLM 이 텍스트 설명만 잘 풀고 graph token 은 noise" 의 ablation 이 더 필요.

## 본 프로젝트 시사점

- **LLM-as-graph-reasoner 패턴의 reference**: 본 프로젝트가 LLM API (Claude/GPT) 를 reasoning 엔진으로 쓸 가능성을 검토할 때 가장 가까운 design.
- **Projector 의 가벼움**: 본 프로젝트의 hybrid 모델 중 가장 light-weight tunable 부분으로 시작 가능 — graph encoder + LLM 모두 frozen, projector 만 학습.
- **차용할 아이디어**: dual-stage tuning (alignment → task-specific) 은 hybrid 학습의 standard recipe 후보. instruction template 의 형식도 본 프로젝트 prompt design 에 그대로 채용 가능.
- **채택하지 않을 부분**: LLM full fine-tune 은 비용 측면에서 본 프로젝트의 초기 단계 부적합 — projector + LoRA 정도로 한정.
- **후속 실험 가설**: GraphGPT 의 zero-shot 우위가 진짜 graph token 덕인지, instruction 의 텍스트 hint 덕인지 ablation — graph token 을 zero vector 로 마스킹했을 때 성능 drop 측정.

## 참고 / 인용

- 공식 코드: <https://github.com/HKUDS/GraphGPT>
- 관련 논문: [[2023-glem-zhao]] (LM-GNN co-training 의 EM 방식), [[2021-graphformers-yang]] (BERT 시대 nested hybrid), [[2021-graphormer-ying]] (graph encoder 의 표준 후보)
- 본 프로젝트 내 인용 위치: 추후 LLM + graph token 실험 노트북에서
