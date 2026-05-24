---
title: "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"
authors: "Xia, M., Gao, T., Zeng, Z., & Chen, D."
year: 2024
venue: "ICLR 2024"
url: "https://arxiv.org/abs/2310.06694"
arxiv_id: "2310.06694"
code_url: "https://github.com/princeton-nlp/LLM-Shearing"
tags: ["computation-graph", "dynamic-param", "resource-aware", "sheared-llama", "structured-pruning", "budget-targeted", "llm"]
status: "draft"
cited_in: []
---

# Sheared LLaMA — Budget-Targeted Structured Pruning

## TL;DR (3줄)

- 사전학습된 LLaMA-7B 를 **사용자가 명시한 target architecture (예: 1.3B, 2.7B)** 로 structured pruning 후 continued pretraining — 동급 size 의 from-scratch 학습 대비 3% compute 로 동등 성능.
- Pruning 자체가 **학습된** (binary mask 와 함께 weight 학습) — Net2Net 류 \"수동 operator\" 와 달리 architecture 결정도 학습 산물.
- 본 프로젝트 관점: **\"학습 중 architecture 수축\" 의 modern reference** — MorphNet 의 budget-aware idea 를 LLM scale 로 끌어올림.

## 핵심 기여

- **Targeted structured pruning**: layer / head / FFN dim / hidden dim 의 4 axes 를 동시에 prune. 각 axis 의 mask 가 학습 가능.
- **Dynamic data loading**: pruning 후 continued pretraining 시 \"잘 학습 안 된 domain\" 의 data 비중 자동 증가 → 적은 compute 로 quality 회복.
- **Budget-targeted**: 정확한 target size (예: 2.7B) 를 hyperparameter 로 명시 → mask 학습이 그 size 로 수렴하도록 constraint loss.
- **From-scratch 대비 3% compute**: LLaMA-7B + 2.7B 까지 prune + 50B token continued pretraining = 동일 2.7B from-scratch 의 3% wallclock.

## 방법 요약

- 데이터: RedPajama (LLaMA pretraining mix), continued pretraining.
- 모델: LLaMA2-7B → Sheared-LLaMA-1.3B / 2.7B.
- 학습:
  1. Source LLaMA-7B 의 모든 layer / head / FFN / hidden dim 에 learnable binary mask 추가
  2. Constraint loss: $\sum (\text{mask})_l \to \text{target FLOPs}$
  3. Joint optimization: pruning loss + LM loss → mask 와 weight 동시 학습
  4. Mask 확정 → 작은 architecture 추출
  5. Continued pretraining 50B token (with dynamic data loading)
- 핵심 흐름 (원논문 §3):

각 axis 의 mask:
$$
z_l \in [0,1]: \quad \mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda \cdot \left( \sum_l z_l \cdot \text{cost}_l - C_{\text{target}} \right)^2
$$

Lagrangian 으로 target cost $C_{\text{target}}$ 강제.

## 실험 / 결과

- Sheared-LLaMA-2.7B vs Pythia-2.8B (from-scratch): 동일 size 에서 일관 우위 — average downstream +3~5%.
- LLaMA-7B 의 13% wallclock 으로 2.7B 도달 vs 100% scratch.
- HellaSwag / ARC / MMLU 등 표준 benchmark 모두 향상.
- 재현성: 공식 PyTorch (Princeton NLP) 공개.

## 한계 / 비판적 시각

- 출발점이 사전학습된 LLaMA 가정 — \"from scratch\" 가 아닌 \"already trained\" 가 필요.
- Continued pretraining 50B token 도 작지 않은 비용.
- Target size 가 미리 정해져야 함 — \"learning the size\" 까지는 아님.
- Mask 학습 시 weight 가 frozen 또는 함께 학습되는지 detail 의존 — hyperparameter 민감.

## 본 프로젝트 시사점

- **Resource constraint + structured pruning 의 modern reference** — MorphNet 의 정신 계승 (budget-aware) + LLM-scale 실용 검증.
- **차용할 아이디어**:
  - **Learnable binary mask + Lagrangian constraint** — \"target size 까지\" 의 명시적 학습 신호. 본 프로젝트의 \"적절한 size 학습\" 결정에 reference.
  - **Dynamic data loading after pruning** — 작은 모델의 한계를 data quality 로 보완 — 본 프로젝트의 \"성장 후 / 수축 후\" continued training 패턴.
  - **4 axes 동시 pruning** — MSG 의 4 axes growth 와 dual 관계. 본 프로젝트가 \"성장 + 수축\" 양방향 다룬다면 axis 정의 reference.
- **채택하지 않을 부분**: \"already trained source\" 가정 — 본 프로젝트는 from-scratch growing 도 가능성 있음. Sheared LLaMA 는 사후 적용.
- **후속 실험 가설**:
  - 본 프로젝트의 growing 방향과 정반대 — \"작은 → 큰\" 대신 \"큰 → 작은\" — 두 방향이 같은 architecture 도달하는지 비교.
  - Lagrangian budget constraint 가 본 프로젝트의 trigger 와 통합 가능한지 — \"plateau + budget 잔여\" 의 결합.

## 참고 / 인용

- 공식 코드: <https://github.com/princeton-nlp/LLM-Shearing> (PyTorch)
- 관련 논문: [MorphNet](2018-morphnet-gordon.md) (origin of budget-aware), [MobileLLM](2024-mobilellm-liu.md) (다른 efficient LLM 접근)
- 본 프로젝트 내 인용 위치: 추후 budget-aware shrinking 실험 노트북에서
