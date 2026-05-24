---
title: "Self-Motivated Growing Neural Network for Adaptive Architecture via Local Structural Plasticity"
authors: "(arXiv preprint; 저자 정보 추후 확정)"
year: 2025
venue: "arXiv preprint (2025-12)"
url: "https://arxiv.org/abs/2512.12713"
arxiv_id: "2512.12713"
tags: ["computation-graph", "dynamic-param", "growing", "structural-plasticity", "function-level", "online-architecture", "rl-controller"]
status: "draft"
cited_in: []
---

# SMGrNN — Local Structural Plasticity 로 학습 중 노드/시냅스 동적 변화

## TL;DR (3줄)

- **Self-Motivated Growing Neural Network (SMGrNN)**: topology 가 학습 중 online 으로 진화하는 신경망 controller. Local Structural Plasticity Module (SPM) 이 뉴런 activation + edge-wise weight update 통계를 짧은 temporal window 로 관찰해서 **뉴런 insertion / pruning** 을 trigger.
- 시냅스 weight 자체는 표준 gradient-based optimizer 로 업데이트. **구조 변경 (structural) + weight 학습 (synaptic) 의 두 차원이 분리**되어 동시 동작.
- RL control task 대비 동등 / 더 높은 return, 낮은 variance, task 에 맞는 network size 자동 조절.

## 핵심 기여

- **Local SPM trigger** — 전역이 아닌 local 관찰 (인접 뉴런 activation + weight update stats) 만으로 노드 추가/제거 결정. scalable.
- **두 차원 분리** — structural change 와 synaptic weight 학습이 독립 메커니즘. structural 은 SPM, synaptic 은 gradient.
- **Online + 자율** — manual architecture tuning 없이 학습 중 capacity 자동 조절.
- **확장성** — Hebbian plasticity / spike-timing-dependent plasticity (STDP) 와 결합 가능한 modular 설계.

## 방법 요약

- 데이터/task: RL control benchmarks
- 모델: MLP-based controller (output: action)
- 학습:
  - 매 step: forward → action → reward → backward (synaptic update)
  - 짧은 temporal window (수십 step) 마다 SPM 평가:
    - 각 neuron 의 activation magnitude / variance 측정
    - 각 edge 의 weight update magnitude 측정
    - 임계 이하 (= "dead") 인 unit prune
    - 임계 이상 (= "active but capacity 부족") 영역에 neuron insert
- 핵심: structural change 가 gradient backward 의 chain rule 을 매 step 다르게 만들지만, **local triggering 이라 학습 stability 유지**.

## 실험 / 결과

- RL 환경 (논문 §4 — control benchmarks).
- MLP baseline 대비 **유사 또는 더 높은 return**, **더 낮은 variance**, **task-appropriate network size** 자동 도달.
- structural plasticity 가 capacity over-provisioning 없이 효율적 학습 가능 입증.

## 한계 / 비판적 시각

- **RL controller (MLP) 만 검증** — Transformer / 대규모 supervised learning 적용 미검증.
- Local trigger 의 threshold 가 hyperparameter — 자동 결정 미해결.
- structural change 시점의 gradient flow 안정성에 대한 이론적 보장 부족 (경험적 안정).
- Hebbian / STDP 결합은 future work 로 명시 — 본 paper 에서는 미구현.

## 본 프로젝트 시사점

> **본 프로젝트의 새 연구 방향 (function-level dynamic) 의 가장 가까운 직접 선례**.

- **적용 가능 부분**:
  - SPM 의 local trigger 패턴 → Transformer 의 function-level 노드 (LayerNorm / Linear / Attention / FFN) 단위로 일반화 가능
  - Phase 1 의 단일 trigger (PlateauTrigger) → SMGrNN 의 multi-criterion local trigger 로 확장
- **차용할 아이디어**:
  - **두 차원 분리** (structural vs synaptic) — Phase 1 의 dead block 문제는 두 차원이 분리 안 되어 발생. SMGrNN 처럼 분리하면 추가 노드의 weight 학습이 structural decision 과 독립.
  - **Local triggering** — Phase 1 의 PlateauTrigger 는 global (loss 만 본다). Local 로 가면 어느 위치에 추가할지 결정 가능.
  - **자율적 prune** — Phase 1 에는 prune 메커니즘 X. dead block 을 자동 제거하는 신호로 활용.
- **채택하지 않을 부분**:
  - RL controller setup — 본 프로젝트는 LM pretraining
  - 단순 MLP — Transformer 의 multi-head attention / residual 의 복잡한 chain rule 에 직접 적용 불가능
- **후속 실험 가설**:
  - Local SPM 을 Transformer block 의 sub-component (qkv / out / fc1 / fc2) 단위로 적용 시 dead block 회피 가능한가?
  - structural change 빈도 (매 N step) vs 학습 stability trade-off — Phase 1 의 PlateauTrigger cooldown=150 보다 짧게 가능한가?
  - prune 메커니즘 도입 시 final n_layers 가 자동으로 task-appropriate 한 값에 수렴하는가?

## 참고 / 인용

- 공식 코드: (arXiv preprint 단계, 코드 공개 여부 추후 확인)
- 관련 논문: [AutoGrow](2020-autogrow-wen.md) (plateau trigger 의 원형), [Net2Net](2016-net2net-chen.md) (function preservation), [DEN](2018-den-yoon.md) (dynamically expandable network)
- 본 프로젝트 내 인용 위치: 새 연구 방향 (function-level dynamic) 의 baseline reference
