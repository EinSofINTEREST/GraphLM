---
title: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
authors: "Elhoushi, M., Shrivastava, A., Liskovich, D., Hosmer, B., Wasti, B., Lai, L., Mahmoud, A., Acun, B., Agarwal, S., Roman, A., Aly, A., et al."
year: 2024
venue: "ACL 2024"
url: "https://arxiv.org/abs/2404.16710"
arxiv_id: "2404.16710"
code_url: "https://github.com/facebookresearch/LayerSkip"
tags: ["computation-graph", "dynamic-param", "deployment", "layerskip", "early-exit", "speculative-decoding", "llm"]
status: "draft"
cited_in: []
---

# LayerSkip — Early-Exit + Self-Speculative Decoding

## TL;DR (3줄)

- 학습 시 **layer dropout + 모든 layer 의 LM head** 학습 → 추론 시 token 마다 early exit (작은 layer 수) 가능.
- Early-exit draft + 전체 모델 verify 의 **self-speculative decoding** — 단일 모델 안에서 speculative 가속.
- 본 프로젝트 관점: **\"학습된 모델에서 token 별 dynamic depth 추론\"** — Mixture of Depths 의 deployment 측 modern dual.

> ⚠️ deployment-time flexibility — 학습 중 동적 증가 자체와는 dual.

## 핵심 기여

- **Layer dropout during training**: 매 batch 마다 일부 layer 를 dropout — 모든 sub-network (early-exit 포함) 가 학습 신호 받음.
- **Early-exit loss**: 모든 layer 끝에 LM head 추가 (또는 shared head) → 작은 layer 수에서도 가능한 prediction.
- **Self-speculative decoding**: early-exit (예: layer 12) 가 draft 생성 → 전체 모델 (layer 32) 이 verify. 단일 모델이라 draft 모델 별도 학습 불필요.
- **1.5~2x inference 가속**: quality 손실 없이 throughput 향상.

## 방법 요약

- 데이터: standard LLaMA pretraining mix.
- 모델: LLaMA-7B / 13B variants — pretraining (또는 continued pretraining) with LayerSkip recipe.
- 학습:
  1. Layer dropout: 매 step 마다 일부 layer skip (probability schedule)
  2. Early-exit loss: 모든 layer $l \in \{l_1, l_2, ..., L\}$ 의 LM head output 의 loss 합
  3. 학습 끝: 모든 early exit point 가 quality 보장
- 핵심 흐름 (원논문 §3):

학습 loss:
$$
\mathcal{L} = \sum_{l \in E} w_l \cdot \mathcal{L}_{\text{LM}}(\text{head}(h_l))
$$

$E$ 는 early-exit 가능 layer 집합, $w_l$ 은 layer 별 가중치 (큰 layer 가중치 더 큼).

## 실험 / 결과

- LLaMA-7B with LayerSkip:
  - Standard inference (32 layer): quality 유지
  - Self-speculative (early-exit 12, verify 32): **1.5x** wallclock 가속
  - Direct early-exit (12 layer 만 사용): quality 약 -5%p
- Self-speculative 가 draft 모델 별도 학습 불필요 — 메모리 이점.
- 재현성: 공식 PyTorch (Meta) 공개.

## 한계 / 비판적 시각

- 학습 비용 증가 — multiple early-exit head + dropout 로 학습 step 시간 ↑.
- Continued pretraining 필요 — 사전학습된 LLaMA 를 그대로 못 쓰고 LayerSkip recipe 로 재학습.
- 1.5x 가속이 \"이상적 환경\" 가정 — 실제 production 의 batching / latency 와 다를 수 있음.
- Quality 손실 없는 sweet spot 은 task-dependent.

## 본 프로젝트 시사점

- **\"학습된 모델의 dynamic depth 추론\" 의 modern reference** — MoD 가 학습 중 dynamic depth 라면 LayerSkip 은 학습된 모델의 추론 시 dynamic depth.
- **차용할 아이디어**:
  - **Layer dropout during training** — 본 프로젝트의 growing model 학습 시 \"여러 depth 가 동시 작동\" 시킬 때 reference. 성장 후 다양 depth 추출의 학습 측 trick.
  - **Multi-exit head** — 본 프로젝트가 성장한 모델의 \"중간 layer 도 의미 있는 prediction\" 능력 부여.
  - **Self-speculative decoding** — 본 프로젝트의 성장된 모델의 추론 가속 옵션. draft 모델 외부 의존 없음.
- **채택하지 않을 부분**: continued pretraining 의 비용 — 본 프로젝트 초기 단계에는 처음부터 LayerSkip recipe 로 학습.
- **후속 실험 가설**:
  - LayerSkip 의 multi-exit 학습이 본 프로젝트의 progressive depth growth 와 동시 가능한지 — 성장 중 모든 depth 가 valid exit point 유지.
  - MoD (학습 중 token-level layer skip) + LayerSkip (학습 후 layer 별 LM head) 의 통합 — 학습/추론 양쪽에서 dynamic depth.

## 참고 / 인용

- 공식 코드: <https://github.com/facebookresearch/LayerSkip> (PyTorch, Meta)
- 관련 논문: [MatFormer](2023-matformer-devvrit.md) (FFN 측 elastic), [Once-for-All](2020-once-for-all-cai.md) (vision 시대), [MoD](2024-mod-raposo.md) (학습 중 dynamic depth)
- 본 프로젝트 내 인용 위치: 추후 early-exit / self-speculative 실험에서
