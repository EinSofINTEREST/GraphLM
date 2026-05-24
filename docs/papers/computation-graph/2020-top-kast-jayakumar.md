---
title: "Top-KAST: Top-K Always Sparse Training"
authors: "Jayakumar, S. M., Pascanu, R., Rae, J. W., Osindero, S., & Elsen, E."
year: 2020
venue: "NeurIPS 2020"
url: "https://arxiv.org/abs/2106.03517"
arxiv_id: "2106.03517"
code_url: ""
tags: ["computation-graph", "dynamic-param", "top-kast", "dst", "sparse", "transformer", "lm", "always-sparse"]
status: "draft"
cited_in: []
---

# Top-KAST — Transformer 시대의 Always-Sparse Training

## TL;DR (3줄)

- DST 의 \"매 step 마다 |weight| 의 top-$k$ 만 forward + 일부 추가 active set 으로 gradient\" 패턴 — RigL 의 batch-level 보다 fine-grained.
- **Transformer LM (Transformer-XL, GPT-2 류) 에 직접 검증** — 기존 DST 가 거의 CNN/MLP-only 였던 한계 돌파.
- 본 프로젝트 관점: **\"Transformer DST 의 표준 reference\"** — RigL 가 vision 위주라면 Top-KAST 는 본 프로젝트 setting (LM) 의 baseline.

## 핵심 기여

- **Always-sparse forward**: 각 step 마다 $|W|$ top-$k$ 만 사용 ($k = \lfloor \text{total} \cdot (1-s) \rfloor$, $s$ = sparsity) (원논문 §2).
- **Wider gradient set**: forward 는 top-$k$, gradient 는 top-$k'$ ($k' > k$) 의 더 큰 \"backward set\" → 비활성 weight 에도 gradient 흘려 \"미래의 활성화\" 신호 제공.
- **Per-step mask update**: RigL 의 epoch-level update 보다 fine-grained.
- **Transformer-XL / GPT-2 적용**: WikiText-103 / WebText 에서 90% sparsity 에서 dense 동등 perplexity 보고.

## 방법 요약

- 데이터: WikiText-103 (Transformer-XL), WebText (GPT-2 류).
- 모델: Transformer-XL, GPT-2 small/medium. Sparsity $s \in \{0.5, 0.7, 0.9, 0.95\}$.
- 학습:
  1. Random sparse init
  2. 각 step:
     - Forward: 현재 |weight| top-$k$ active set 만 사용
     - Backward: 더 큰 top-$k'$ set 에 gradient 흘림 (비활성 weight 도 update)
     - 마스크 자동 update — 다음 step 의 top-$k$ 가 자연스럽게 변동
- 핵심 흐름 (원논문 §2, forward/backward mask):

각 weight matrix $W$ 의 active mask:
$$
\mathcal{M}_W^{\text{forward}} = \{(i,j) : |W_{ij}| \in \text{top-}k\}, \quad
\mathcal{M}_W^{\text{backward}} = \{(i,j) : |W_{ij}| \in \text{top-}k'\}
$$

$k' > k$ 로 두 mask 가 nested.

## 실험 / 결과

- Transformer-XL WikiText-103 90% sparsity: perplexity 23.9 (dense baseline 23.7 — 거의 동등).
- GPT-2 small WebText 80% sparsity: 동등 perplexity.
- 95% sparsity 에서도 가용 — extreme sparse 영역 확장.
- 재현성: DeepMind 의 공식 구현 비공개 (논문 detail 만), 후속 비공식 PyTorch 재구현 다수.

## 한계 / 비판적 시각

- **공식 코드 미공개** — 재현에 detail 부족.
- Backward set $k'$ 의 hyperparameter — sweet spot 가 dataset / 모델 의존.
- RigL 의 explicit prune-grow 와 달리 \"항상 sparse\" 가정 → mask transition 의 명시적 제어가 약함.
- 큰 LM (1B+) scale 검증 미보고 — 후속 작업 필요.

## 본 프로젝트 시사점

- **본 프로젝트 패러다임의 Transformer DST reference** — RigL 가 vision 위주의 한계를 Top-KAST 가 보완. 본 프로젝트가 sparse 보조 모듈을 LM 에 직접 적용한다면 reference 1순위.
- **차용할 아이디어**:
  - **Forward vs backward mask 분리** — backward set 이 더 크다 = \"미래 활성화 후보\" 가 gradient 신호 받음. 본 프로젝트의 sparse + growth 결합 시 이 패턴 유용.
  - **Per-step mask update** — RigL 의 epoch-level 보다 빠른 adaptation. 작은 dataset 의 toy 실험에서 더 빨리 sparse pattern 수렴.
  - **Transformer 의 layer 별 sparsity 차등** — 모든 layer 균등이 아니라 layer 별 다른 $s$ 가능.
- **채택하지 않을 부분**: 공식 코드 미공개로 detail 우회 필요. 후속 비공식 구현 활용.
- **후속 실험 가설**:
  - Top-KAST 의 always-sparse 와 본 프로젝트의 growing 가 결합 가능한지 — \"sparse + 새 connection 추가\" 의 통합 framework.
  - $k' / k$ 비율의 작은 LM 에서의 sweet spot — 너무 크면 noise, 너무 작으면 stagnation.

## 참고 / 인용

- 공식 코드: 미공개 (DeepMind 내부)
- 관련 논문: [RigL](2020-rigl-evci.md) (vision DST 선례), [SET](2018-set-mocanu.md) (DST origin)
- 본 프로젝트 내 인용 위치: 추후 Transformer DST 실험 노트북에서
