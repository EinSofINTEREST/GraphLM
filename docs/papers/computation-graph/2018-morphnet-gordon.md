---
title: "MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks"
authors: "Gordon, A., Eban, E., Nachum, O., Chen, B., Wu, H., Yang, T.-J., & Choi, E."
year: 2018
venue: "CVPR 2018"
url: "https://arxiv.org/abs/1711.06798"
arxiv_id: "1711.06798"
code_url: "https://github.com/google-research/morph-net"
tags: ["computation-graph", "dynamic-param", "morphnet", "resource-aware", "morphism", "flops-constraint", "shrink-grow"]
status: "draft"
cited_in: []
---

# MorphNet — Resource-Constrained Architecture Morphism

## TL;DR (3줄)

- 네트워크 구조 학습을 **사용자가 명시한 resource budget (FLOPs / memory)** 하에서 진행 — sparse regularization 으로 shrink, 그 후 uniform multiplier 로 grow 의 단순 cycle.
- BatchNorm 의 $\gamma$ 가중치를 \"width proxy\" 로 활용 — group sparsity 로 작은 $\gamma$ 의 channel 제거 → 자동 architecture morphism.
- 본 프로젝트 관점: **\"성장이 무한이 아니라 budget 제약 하\"** 의 reference. AutoGrow / Firefly 의 trigger 가 \"언제 + 어디\" 를 결정한다면, MorphNet 은 \"얼마까지\" 를 결정.

## 핵심 기여

- **Shrink-Expand 의 2-step framework**:
  1. **Shrink**: BatchNorm $\gamma$ 에 group lasso penalty → 작은 channel 자동 prune
  2. **Expand**: 남은 architecture 의 모든 dim 을 uniform multiplier $\alpha$ 로 확대해 budget 채움
- **Budget-aware**: \"FLOPs ≤ X\" 또는 \"latency ≤ Y\" 의 명시적 제약 하 architecture 학습.
- **BatchNorm-based proxy**: 별도 mask 변수 없이 BN $\gamma$ 자체를 \"channel 중요도\" 로 활용 — 구현 단순.
- **Single-pass training**: NAS 류의 search-then-train 이 아니라 한 번 학습으로 architecture 결정.

## 방법 요약

- 데이터: ImageNet (Inception V2, MobileNet, ResNet 변형).
- 모델: 위 backbone 들. budget: 모델 별 FLOPs 또는 latency 명시.
- 학습:
  1. Standard SGD + 추가 group-lasso penalty on BN $\gamma$
  2. $\gamma$ 가 threshold 이하인 channel/layer 제거 (shrink)
  3. 남은 architecture 의 width 를 multiplier $\alpha$ 로 grow — budget 안 채울 때까지
  4. 결과 architecture 로 standard 재학습 (또는 fine-tune)
- 핵심 흐름:

각 BN layer 의 group lasso:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \sum_l \frac{\|\gamma_l\|_1}{\text{FLOPs}_l}
$$

FLOPs 가 큰 layer 일수록 더 강한 penalty (resource-aware).

## 실험 / 결과

- ImageNet MobileNet: 동일 FLOPs 에서 hand-tuned baseline 대비 +1.3% top-1.
- ResNet-101: FLOPs 50% 절감하면서 top-1 유지.
- 단일 학습으로 architecture 자동 발견 — 별도 NAS search 없음.
- 재현성: 공식 TensorFlow (Google MorphNet) 공개.

## 한계 / 비판적 시각

- **CNN/BatchNorm 의존** — Transformer (LayerNorm) 에 직접 적용 어려움. LayerNorm 의 $\gamma$ 는 BN 과 다른 정규화 의미.
- Shrink-then-Expand 의 2-pass — 학습 효율 측면에서 single-pass adaptive growth (Firefly, GradMax) 대비 비효율.
- Budget 이 hyperparameter — 적절한 FLOPs target 은 도메인 지식 필요.
- Expand 의 uniform multiplier $\alpha$ — 모든 axis 균등 확장은 suboptimal (CompoundGrowth 류가 보완).

## 본 프로젝트 시사점

- **Budget-aware 성장의 reference** — 본 프로젝트가 \"학습 중 모델을 키우되 메모리 / 추론 비용은 X 이하\" 제약을 둔다면 MorphNet 의 group lasso + budget 가중치 패턴이 직접 적용.
- **차용할 아이디어**:
  - **Resource-weighted regularization** — FLOPs 가 큰 layer 일수록 penalty 강화 → 효율적 channel 만 살아남음. Transformer 의 FFN/attention 비용 차이 반영 가능.
  - **Shrink-Expand 2-step** — single growth operator (LiGO) 대비 shrink 도 포함 → 과확장 자동 회복. 본 프로젝트의 \"over-growth\" 안전망.
- **채택하지 않을 부분**:
  - BatchNorm $\gamma$ proxy — Transformer 에는 부적합. 대안: attention head 의 norm, FFN expert 의 utilization 등 Transformer 특화 proxy 필요.
  - Single-pass 가정 — 본 프로젝트는 학습 중 여러 번 grow/shrink 가능성 큼.
- **후속 실험 가설**:
  - LayerNorm 의 $\gamma$ 가 BN 만큼 channel 중요도 신호로 작동하는지 — 작동 안 하면 attention head 의 output norm 으로 대체 가능성.
  - Firefly 의 \"어디를 키울지\" + MorphNet 의 \"얼마나 budget 까지\" 합성 — 두 직교 dimension 의 결합 효과.

## 참고 / 인용

- 공식 코드: <https://github.com/google-research/morph-net> (TensorFlow)
- 관련 논문: [AutoGrow](2020-autogrow-wen.md) (다른 종류 자동 architecture 결정), [Once-for-All](2020-once-for-all-cai.md) (deployment 시점 적응), [GradMax](2022-gradmax-evci.md) (gradient-based 다른 시각)
- 본 프로젝트 내 인용 위치: 추후 budget-aware growing Transformer 실험 노트북에서
