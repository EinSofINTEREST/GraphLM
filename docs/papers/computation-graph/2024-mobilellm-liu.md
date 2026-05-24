---
title: "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"
authors: "Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., Lai, L., & Chandra, V."
year: 2024
venue: "ICML 2024"
url: "https://arxiv.org/abs/2402.14905"
arxiv_id: "2402.14905"
code_url: "https://github.com/facebookresearch/MobileLLM"
tags: ["computation-graph", "dynamic-param", "resource-aware", "mobilellm", "sub-billion", "on-device", "efficient-design"]
status: "draft"
cited_in: []
---

# MobileLLM — Sub-Billion 효율 LLM Design

## TL;DR (3줄)

- On-device 배포를 위한 sub-1B LLM 의 **architecture design choice 체계 정리** — depth (deep & thin) vs width 의 trade-off, embedding sharing, grouped query attention, block-wise weight sharing 등.
- MobileLLM-125M / 350M 이 동일 size 의 기존 baseline (OPT-125M, OPT-350M 등) 대비 zero-shot benchmark 평균 +2~3%p — 가장 efficient sub-1B LLM.
- 본 프로젝트 관점: **\"고정된 budget 내 최적 design\"** 의 reference. \"성장의 종착점이 어디여야 하는가\" 의 답.

## 핵심 기여

- **Deep & thin architecture**: 같은 param 수에서 wide-shallow 보다 deep-narrow 가 zero-shot quality 우위 — 30-layer × 576 hidden 이 12-layer × 1024 hidden 보다 좋음.
- **Embedding-LM head sharing**: input embedding 과 output projection 의 weight 공유 — sub-billion 영역에서 quality 영향 작고 효율 큼.
- **Grouped Query Attention (GQA)**: heads 의 KV projection 공유 — Llama2 의 기법을 sub-1B 에 적용.
- **Block-wise weight sharing**: 인접 layer 의 weight 공유로 추가 효율 — MobileLLM-LS 변형.

## 방법 요약

- 데이터: SlimPajama (LLaMA pretraining-like corpus), 1T token.
- 모델: 125M / 350M 변형. 30-32 layer, 576-960 hidden, 9-15 head, GQA, embedding sharing.
- 학습:
  1. 표준 causal LM pretraining (1T token)
  2. Architecture ablation 으로 design choice 결정
  3. Optionally block-wise sharing 추가 (LS 변형)
- 핵심 발견 (원논문 §3 ablation):

\"같은 param budget 에서:
- Depth > Width
- Embedding sharing 효과 큰 영향 작음
- GQA 효율 ↑ quality 거의 동등\"

## 실험 / 결과

- MobileLLM-125M zero-shot 평균: 46.3 vs OPT-125M 36.5 (+9.8%p).
- MobileLLM-350M zero-shot 평균: 49.8 vs OPT-350M 41.4 (+8.4%p).
- BoolQ / PIQA / SIQA / HellaSwag / WinoGrande / ARC-e/c / OBQA — 거의 모든 task 에서 우위.
- 재현성: 공식 PyTorch (Meta) 공개.

## 한계 / 비판적 시각

- 1T token pretraining 비용 여전 큼 — 작은 lab 부담.
- **고정 architecture** 검증 — 본 프로젝트의 \"학습 중 동적\" 와는 직접 반대 패러다임. Design choice insight 만 transferable.
- Sub-1B 영역 위주 — 1B+ 의 design choice 는 다를 수 있음 (실제로 1B+ 는 wide 가 우위 보고).
- Block-wise sharing 의 inference latency 증가 — memory ↓ but compute ↓ X.

## 본 프로젝트 시사점

- **\"고정 budget 내 최적 design\" 의 reference** — 본 프로젝트의 성장이 어디서 멈춰야 하는지의 sweet spot 정보. \"125M 까지는 deep-narrow, 1B+ 부터는 wide\" 의 design rule 적용.
- **차용할 아이디어**:
  - **Deep > Width 의 sub-1B 경험적 우위** — 본 프로젝트의 toy 실험이 sub-1B 영역에서 진행된다면 depth 우선 성장.
  - **Embedding sharing** — 본 프로젝트의 작은 model 구현 시 default 채택 (quality 손실 작고 메모리 절감 큼).
  - **GQA** — head 수 성장 시 KV 공유로 효율 유지.
- **채택하지 않을 부분**: block-wise weight sharing — 본 프로젝트의 \"layer 별 독립 성장\" 과 충돌. 단 메모리 극단 부족 시 fallback 옵션.
- **후속 실험 가설**:
  - 본 프로젝트의 \"성장 trajectory\" 가 MobileLLM 의 deep-narrow recipe 와 자연 수렴하는지 — adaptive trigger 가 자동으로 depth 성장 선호 패턴 보이는지.
  - GQA 와 grouped FFN expert 의 결합 — sub-1B 영역의 추가 efficiency.

## 참고 / 인용

- 공식 코드: <https://github.com/facebookresearch/MobileLLM> (PyTorch, Meta)
- 관련 논문: [Sheared LLaMA](2024-sheared-llama-xia.md) (다른 budget-aware 접근), [MorphNet](2018-morphnet-gordon.md) (origin)
- 본 프로젝트 내 인용 위치: 추후 sub-1B target 실험 노트북에서
