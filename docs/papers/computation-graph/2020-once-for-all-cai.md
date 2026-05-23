---
title: "Once-for-All: Train One Network and Specialize it for Efficient Deployment"
authors: "Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S."
year: 2020
venue: "ICLR 2020"
url: "https://arxiv.org/abs/1908.09791"
arxiv_id: "1908.09791"
code_url: "https://github.com/mit-han-lab/once-for-all"
tags: ["computation-graph", "dynamic-param", "ofa", "once-for-all", "supernet", "deployment", "elastic", "progressive-shrinking"]
status: "draft"
cited_in: []
---

# Once-for-All — Train Once, Deploy Many

## TL;DR (3줄)

- 단일 supernet 학습 한 번으로 다양한 size 의 sub-network 들을 **직접 추출 가능** — 별도 retrain 없이 deployment 시점에서 device 별 (latency / memory) 최적 architecture 선택.
- 핵심: **progressive shrinking** training schedule — depth → kernel → width → resolution 순으로 elastic dim 확장.
- 본 프로젝트 관점: **\"학습 중 dynamic param count\" 의 dual** — 학습 중에는 다양한 size 가 supernet 안에 공존, 학습 후에는 어느 size 든 추출 가능. \"성장\" 이 아니라 \"이미 다양\".

## 핵심 기여

- **Single supernet, many sub-networks**: $10^{19}$ 개 sub-network 조합이 모두 학습 완료 후 fine-tune 없이 사용 가능.
- **Progressive shrinking**:
  1. Largest sub-network 만 학습 (warm start)
  2. Elastic depth 추가 (sub-network 가 다양한 depth 선택)
  3. Elastic kernel size
  4. Elastic width
  5. Elastic resolution
- **Knowledge distillation**: 학습 중 큰 sub-network 가 작은 sub-network 의 teacher 로 작용.
- **Deployment-time NAS**: device 별 search (latency / FLOPs 제약) 가 신경망 학습이 아니라 단순 sub-network 선택으로 축소.

## 방법 요약

- 데이터: ImageNet.
- 모델: MobileNetV3 기반 supernet — depth $\in \{2,3,4\}$, kernel $\in \{3,5,7\}$, width 다양, resolution 다양.
- 학습:
  1. Largest sub-network warm-up
  2. Progressive shrinking: stage 마다 한 elastic dim 추가 → 그 dim 의 모든 변형 sample
  3. Distillation: 작은 sub-network 가 큰 sub-network 의 output 으로 학습
  4. 약 1200 GPU-hour (Tesla V100, 한 번에 supernet 학습)
- 핵심 아이디어 (deployment, 원논문 §3.4):

각 target device 마다:
$$
\text{best\_subnet}(d) = \arg\max_{s \in \text{subnets}} \text{accuracy}(s) \;\text{s.t.}\; \text{latency}(s, d) \le L_d
$$

학습된 latency predictor + accuracy predictor 로 $10^{19}$ 후보 빠르게 검색.

## 실험 / 결과

- ImageNet: OFA-Large 79.0% top-1 (동일 FLOPs 의 MobileNetV3 75.2%).
- 50개 device 의 latency-constrained accuracy 가 device 별 separate training 동등 (1/200 비용).
- Deployment 시점 device 별 search: 분 단위.
- 재현성: 공식 PyTorch 공개.

## 한계 / 비판적 시각

- **Supernet 학습 비용** — 1200 GPU-hour, 작은 lab 에 부담.
- 학습된 sub-network 의 quality 가 같은 architecture 의 from-scratch 학습 대비 약간 열위 (지식 distillation 가 보완).
- **CNN/Vision 위주** — Transformer / LM 의 OFA 변형은 AutoFormer 가 일부 시도.
- Progressive shrinking 의 elastic dim 순서가 hyperparameter — 변경 시 성능 차이.

## 본 프로젝트 시사점

- **\"학습 중 dynamic param count\" 의 dual interpretation** — OFA 는 학습 중에 \"성장\" 이 아니라 \"여러 size 가 공존\". 본 프로젝트의 \"trigger 기반 성장\" 과는 다른 패러다임이지만, **deployment 측면의 supernet 패턴** 이 본 프로젝트의 후속 단계에서 유용.
- **차용할 아이디어**:
  - **Progressive shrinking schedule** — 큰 모델부터 학습 → 작은 sub-network 로 distill. 본 프로젝트의 \"성장한 모델에서 작은 sub-network 추출\" 기능에 reference.
  - **Architecture-level distillation** — 큰 → 작은 transfer 가 학습 가속.
- **채택하지 않을 부분**:
  - Supernet 전체 학습의 비용 — 본 프로젝트 초기 단계에는 부담.
  - Vision-specific elastic dim (resolution / kernel) — Transformer 의 head / FFN dim 으로 대체.
- **후속 실험 가설**:
  - 본 프로젝트의 \"성장된 큰 모델\" 에서 OFA 식 progressive shrinking 으로 작은 deployment-ready sub-network 추출 — 학습 끝 시점의 dual operator.
  - Adaptive growth + OFA 의 \"학습 중 성장 → 학습 후 적응적 축소\" hybrid framework.

## 참고 / 인용

- 공식 코드: <https://github.com/mit-han-lab/once-for-all> (PyTorch)
- 관련 논문: [MorphNet](2018-morphnet-gordon.md) (다른 resource-aware 접근), [AutoFormer](2021-autoformer-chen.md) (Transformer 의 supernet)
- 본 프로젝트 내 인용 위치: 추후 deployment-time flexibility 실험 노트북에서
