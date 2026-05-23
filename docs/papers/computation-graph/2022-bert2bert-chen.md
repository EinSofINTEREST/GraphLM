---
title: "bert2BERT: Towards Reusable Pretrained Language Models"
authors: "Chen, C., Yin, Y., Shang, L., Jiang, X., Qin, Y., Wang, F., Wang, Z., Chen, X., Liu, Z., & Liu, Q."
year: 2022
venue: "ACL 2022"
url: "https://arxiv.org/abs/2110.07143"
arxiv_id: "2110.07143"
code_url: "https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/bert2BERT"
tags: ["computation-graph", "dynamic-param", "bert2bert", "growth", "function-preserving", "model-expansion", "lm"]
status: "draft"
cited_in: []
---

# bert2BERT — Function-Preserving Expansion for BERT

## TL;DR (3줄)

- 사전학습된 BERT-Base 의 weight 를 **function-preserving 하게 BERT-Large 로 mapping** 해, 처음부터 BERT-Large 학습 대비 **사전학습 비용 ~45% 절감**.
- Net2Net 의 CNN 아이디어를 Transformer 의 attention + LayerNorm + residual 에 맞게 일반화 — width 와 depth 동시 확장 지원.
- 본 프로젝트 패러다임의 **BERT 도메인 modern reference** — 큰 모델을 \"처음부터\" 가 아니라 \"작은 모델로 시작해 키운다\" 패러다임의 실용적 증명.

## 핵심 기여

- **FPI (Function-Preserving Initialization)** for Transformer:
  - **Width expansion**: hidden dim / head 수 / FFN intermediate dim 의 확장. 새 차원은 기존 차원의 복제 또는 zero-pad + scale.
  - **Depth expansion**: 새 layer 를 identity-like init (residual scale 0 또는 layer 복사).
- **Stage-wise expansion**: BERT-Base (12-layer, 768 hidden) → BERT-Large (24-layer, 1024 hidden) 의 단계적 확장. 각 stage 후 추가 학습.
- **AKI (Advanced Knowledge Initialization)**: FPI 의 \"strict\" function preservation 을 약간 완화 — 작은 perturbation 으로 student 가 teacher 의 local minimum 에서 벗어날 여지 → 더 나은 final quality.
- **GLUE / SQuAD 등 다운스트림에서 처음부터 학습된 BERT-Large 동등 또는 약간 우위**.

## 방법 요약

- 데이터: BookCorpus + English Wikipedia (BERT 표준 사전학습 corpus), GLUE / SQuAD (다운스트림).
- 모델: BERT-Base → BERT-Large mapping.
- 학습:
  1. 사전학습된 BERT-Base 확보
  2. FPI 로 width/depth 확장 — BERT-Large 의 weight 가 BERT-Base 와 동일 output 보장 상태에서 시작
  3. 추가 사전학습 (BookCorpus + Wikipedia)
  4. AKI 변형: FPI 후 작은 random perturbation 으로 strict preservation 해제
- 핵심 흐름 (depth expansion):

새 layer $L_{\text{new}}$ 를 두 가지 방식 중 하나로 init:
- **Identity init**: residual scale 0 → $h_{\text{out}} = h_{\text{in}}$
- **Copy init**: 가장 가까운 기존 layer 복사 → 같은 변환 수행

이후 strict function preservation 보장. AKI 는 + small noise 로 변형.

## 실험 / 결과

- BERT-Large 사전학습: 처음부터 학습 대비 **45% 시간 절감** (동일 perplexity 도달).
- GLUE average: bert2BERT-Large 84.4 vs 처음부터 학습된 BERT-Large 84.0 — 약간 우위.
- SQuAD v1.1 F1: 90.8 (동등).
- Net2Net 기반 baseline 대비도 우위 (Net2Net 직접 적용은 BERT 의 LayerNorm + residual 에 suboptimal).
- 재현성: 공식 PyTorch (Huawei Noah) 공개.

## 한계 / 비판적 시각

- BERT-Base → BERT-Large 라는 **특정 pair** 검증 — 임의 크기 확장의 generality 미보고.
- Width 확장의 weight 복제 + scaling 이 hyperparameter (어느 dim 을 복제할지) 가짐.
- LayerNorm 자체는 running statistics 가 없지만, 확장 후 LayerNorm 의 affine 파라미터 ($\gamma, \beta$) 와 residual scale init 의 compatibility 처리에 대한 명확화 부족 — student 의 layer 별 normalization scale 이 teacher 와 다르면 function preservation 깨질 수 있음.
- LLM-scale (1B+) 미검증 — 후속 LiGO / MSG 가 확장.

## 본 프로젝트 시사점

- **Transformer 도메인 growing 의 표준 reference** — 본 프로젝트가 작은 GraphLM → 큰 GraphLM 으로 확장하는 모듈을 구현할 때 FPI 의 Transformer 특화 트릭이 직접 적용.
- **차용할 아이디어**:
  - **FPI 의 Transformer-specific width/depth recipe** — multi-head attention 의 head 추가 / FFN intermediate dim 확장 / LayerNorm 복제 등의 구체적 패턴.
  - **AKI 의 small perturbation** — strict preservation 의 학습 정체 우려를 깨는 실용 트릭. 본 프로젝트의 확장 직후 noise 주입 default.
  - **Stage-wise schedule** — 한 번에 큰 확장이 아니라 작은 확장 여러 번 — Progressive Stacking 과 결합 가능.
- **채택하지 않을 부분**: BERT-specific 의 special token / segment embedding 처리 — decoder LM 또는 다른 GraphLM 구조에는 별도 적용 필요.
- **후속 실험 가설**:
  - FPI strict vs AKI 의 작은 dataset 에서 quality / 학습 안정성 차이 측정.
  - bert2BERT 의 stage-wise expansion 과 Stacking 의 progressive growth 의 합성 — 두 단계가 직교적으로 효율 누적되는지.

## 참고 / 인용

- 공식 코드: <https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/bert2BERT> (PyTorch)
- 관련 논문: [Net2Net](2016-net2net-chen.md) (CNN 시대 origin), [Progressive Stacking](2019-stacking-gong.md) (BERT 의 다른 growth 방향), [LiGO](2023-ligo-wang.md) (학습 가능 일반화)
- 본 프로젝트 내 인용 위치: 추후 BERT-style growing 실험 노트북에서
