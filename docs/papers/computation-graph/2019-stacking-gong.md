---
title: "Efficient Training of BERT by Progressively Stacking"
authors: "Gong, L., He, D., Li, Z., Qin, T., Wang, L., & Liu, T.-Y."
year: 2019
venue: "ICML 2019"
url: "https://proceedings.mlr.press/v97/gong19a.html"
arxiv_id: "1905.05759"
code_url: "https://github.com/gonglinyuan/StackingBERT"
tags: ["computation-graph", "dynamic-param", "stacking", "bert", "progressive-growth", "depth-growth", "lm"]
status: "draft"
cited_in: []
---

# Progressive Stacking BERT — Transformer 도메인의 첫 Growth Recipe

## TL;DR (3줄)

- BERT 사전학습을 **얕은 모델 (예: 3-layer) 부터 시작 → 학습 진행 중 layer 를 점진적으로 복사·추가** 해 최종 깊이 (12-layer) 까지 성장시키는 progressive training.
- 같은 quality 의 BERT-Base 도달까지 약 **25% 학습 시간 절감** + 최종 perplexity 동등 또는 약간 우위.
- 본 프로젝트 패러다임의 **Transformer 도메인 첫 직접 적용 사례**. \"학습 step 마다 모델이 크기를 키운다\" 의 BERT 시대 시초.

## 핵심 기여

- **Layer duplication-based stacking**: 학습된 $N$-layer BERT 의 상위 절반 layer 를 복사해 $2N$-layer 로 확장. 복사된 layer 는 동일 weight 로 시작 → 작은 perturbation 후 학습 계속.
- **Multi-stage progressive schedule**: BERT-Base 12-layer 의 경우 3 → 6 → 12 layer 의 3-stage. 각 stage 의 step 수는 hyperparameter.
- **Function preservation 의 BERT 적응**: Net2DeeperNet 의 identity init 과 달리, BERT 의 self-attention + LayerNorm 조합에는 layer 복사가 더 안정.
- **Empirical recipe**: 어느 stage 에서 어느 layer 를 복사할지의 best practice — 상위 layer 복사가 하위보다 우위 (\"학습이 더 된\" layer 의 representation 활용).

## 방법 요약

- 데이터: BookCorpus + English Wikipedia (BERT 표준 사전학습 corpus).
- 모델: BERT-Base (12-layer, 768 hidden) target. 3 → 6 → 12 layer progressive.
- 학습:
  1. 3-layer BERT 학습 (전체 step 의 일부)
  2. 상위 절반 (1.5 ≈ 2 layer) 을 복사해 6-layer 로 stack
  3. 다시 학습 진행 후 6 → 12 layer 로 확장
  4. 최종 단계는 표준 BERT 학습 schedule
- 핵심 흐름:

각 stacking step 에서:
$$
\text{Layer}^{(N+1, \ldots, 2N)} \leftarrow \text{copy}\,\text{Layer}^{(N/2+1, \ldots, N)}
$$

상위 절반 복사가 \"이미 학습된\" representation 을 새 layer 에 활용.

## 실험 / 결과

- BERT-Base 사전학습 시간: 기존 방식 대비 **약 25% 단축** (동일 perplexity 도달).
- 다운스트림 GLUE / SQuAD: 동등 또는 약간 우위.
- 학습 곡선 분석: stacking 직후 일시적 perplexity spike (작은 폭) → 빠른 복구.
- 재현성: 공식 PyTorch 공개. 후속 bert2BERT / LiGO 의 baseline 으로 자주 비교.

## 한계 / 비판적 시각

- **복사 기반의 단순함** — 학습 가능한 expansion operator 가 아니라 weight 복제만 → suboptimal 가능성 (후속 LiGO 가 학습 가능 mapping 으로 개선).
- Stacking timing (몇 step 에 확장할지) 의 hyperparameter 민감 — 자동 결정 미해결.
- 폭 (hidden dim) 확장은 미포함 — depth 만 다룸. CompoundGrowth (Gu et al., 2021) 가 폭/깊이 동시 보완.
- BERT-Large 등 더 큰 모델 / 다른 architecture (decoder LM) 확장 미보고.

## 본 프로젝트 시사점

- **본 프로젝트의 첫 growing Transformer 구현 reference** — Stacking 의 단순함이 PyTorch 구현 비용을 크게 낮춤. \"layer 복사\" 만으로도 의미 있는 학습 효율 달성.
- **차용할 아이디어**:
  - **Progressive depth schedule** — 본 프로젝트의 `src/graphlm/training/` 의 학습 loop 에 stage 별 model size 변경 callback.
  - **상위 layer 복사 우선** — 어느 layer 를 복사할지의 design choice. 실험 변수로 노출.
  - **Function-not-strictly-preserving** 의 실용성 — perfect identity 가 아니라도 \"잠시 spike 후 복구\" 가 받아들일 만함.
- **채택하지 않을 부분**: encoder-only BERT 특화 schedule — decoder LM 의 경우 다른 schedule 필요할 수 있음 (별도 실험).
- **후속 실험 가설**:
  - Stacking 시점의 \"warm restart\" 가 학습 곡선의 spike 를 얼마나 키우는지 / 복구 시간이 모델 크기에 어떻게 비례하는지.
  - 상위 vs 하위 layer 복사의 영향 — 본 논문 주장 (\"상위 우위\") 의 작은 모델에서의 재현.

## 참고 / 인용

- 공식 코드: <https://github.com/gonglinyuan/StackingBERT> (PyTorch)
- 관련 논문: [Net2Net](2016-net2net-chen.md) (CNN 시대 origin), [bert2BERT](2022-bert2bert-chen.md) (BERT-Base → BERT-Large mapping), [LiGO](2023-ligo-wang.md) (학습 가능 일반화), [MSG](2024-msg-yao.md) (LLM 스케일)
- 본 프로젝트 내 인용 위치: 추후 progressive growth 첫 실험 노트북에서
