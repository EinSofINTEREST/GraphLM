---
title: "Modular Deep Learning"
authors: "Pfeiffer, J., Ruder, S., Vulić, I., & Ponti, E. M."
year: 2023
venue: "TMLR 2023"
url: "https://arxiv.org/abs/2302.11529"
arxiv_id: "2302.11529"
tags: ["computation-graph", "modular", "survey", "routing", "aggregation", "function-level", "transformer"]
status: "draft"
cited_in: []
---

# Modular Deep Learning — Modular Methods 의 종합 Survey

## TL;DR (3줄)

- Deep learning 의 **modular methods 종합 survey** — modules / routing / aggregation 의 3 차원 분류로 기존 연구 통합.
- Module = locally updatable 한 parameter set (e.g. adapter, prompt, LoRA), Routing = 어느 module 을 활성화할지 결정, Aggregation = module outputs 합성 방법.
- positive transfer / compositionality / parameter efficiency 의 modular 의 3 핵심 장점 정리. function-level dynamic 컨셉의 이론적 framework.

## 핵심 기여

- **3 차원 taxonomy**: Modules (parameter 결합 방식), Routing (활성화 결정), Aggregation (output 합성).
- 기존 다양한 modular method (adapter, LoRA, MoE, conditional computation, mixture of softmax 등) 를 통합 framework 로 분류.
- Modular architecture 의 학습 dynamics, transfer learning, multi-task, continual learning 측면에서의 advantage 정리.

## 방법 요약

- Survey 논문 — 자체 새 method 제시는 X. 기존 method 의 framework 화.
- **Module 종류**: parameter composition (LoRA, additive), input composition (adapter), function composition (bottleneck), hyper-network composition.
- **Routing 종류**: fixed (hard-coded), learned hard (top-k), learned soft (weighted sum).
- **Aggregation 종류**: parameter sum, output sum, attention.

## 실험 / 결과

- Survey — 본 paper 의 실험은 X. 다양한 modular method 의 trade-off 표 / 비교 정리.

## 한계 / 비판적 시각

- Survey 라 specific implementation guidance 부족.
- **Static modular structure** — 학습 중 module 의 추가/제거 (structural plasticity) 는 다루지 않음.
- 본 프로젝트의 "training-time dynamic" 패러다임은 본 survey 의 범위 밖 (대부분 fixed modular).

## 본 프로젝트 시사점

> **본 프로젝트의 function-level dynamic 컨셉의 이론적 framework reference**.

- **적용 가능 부분**:
  - 사용자 컨셉의 "함수 단위 노드" 를 Pfeiffer survey 의 module 정의로 다시 정리 — modules + routing + aggregation 의 3 차원 명세
  - Phase 1 의 GrowingDecoder = "fixed routing + fixed aggregation + dynamic modules" — 본 survey 의 분류에서 어디인지 명확
  - 사용자 컨셉 = "dynamic routing + dynamic aggregation + dynamic modules" — 3 차원 모두 학습 중 변화
- **차용할 아이디어**:
  - **Function composition** module 패턴 (transformer 의 bottleneck adapter 와 같은 구조) → 추가 노드 init 의 후보
  - **Soft routing** (학습 가능 weighted sum) → 사용자 컨셉의 dynamic synapse 의 부드러운 구현
- **채택하지 않을 부분**:
  - Adapter / LoRA 의 parameter-efficient fine-tuning 패턴 — 본 프로젝트는 from-scratch growing
- **후속 실험 가설**:
  - Pfeiffer 의 3 차원 framework 안에서 본 프로젝트의 GrowingDecoder + 사용자 컨셉을 정확히 위치시킬 수 있는가?
  - Routing 의 fixed → soft → hard 의 단계적 dynamic 화가 학습 stability 에 미치는 영향

## 참고 / 인용

- 공식 web: <https://www.ruder.io/modular-deep-learning/>
- 관련 논문: [Are Neural Nets Modular?](2021-modular-transformer-csordas.md), [Modular Networks](2018-modular-networks-kirsch.md), [MoE](2017-moe-shazeer.md)
- 본 프로젝트 내 인용 위치: function-level dynamic 컨셉의 이론적 framework 정리
