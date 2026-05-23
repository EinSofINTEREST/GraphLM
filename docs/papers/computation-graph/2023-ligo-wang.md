---
title: "Learning to Grow Pretrained Models for Efficient Transformer Training"
authors: "Wang, P., Panda, R., Hennigen, L. T., Greengard, P., Karlinsky, L., Feris, R., Cox, D., Wang, Z., & Kim, Y."
year: 2023
venue: "ICLR 2023"
url: "https://arxiv.org/abs/2303.00980"
arxiv_id: "2303.00980"
code_url: "https://github.com/VITA-Group/LiGO"
tags: ["computation-graph", "dynamic-param", "ligo", "growing", "learnable", "linear-operator", "transformer", "modern"]
status: "draft"
cited_in: []
---

# LiGO — Learnable Linear Growth Operators

## TL;DR (3줄)

- 작은 사전학습 Transformer 의 weight 를 큰 Transformer 로 mapping 하는 **학습 가능한 linear operator** 학습 — 기존 heuristic (Net2Net / bert2BERT 의 weight 복제 + scaling) 의 일반화.
- Width / depth 의 두 종류 expansion 을 **decomposed Kronecker product** 로 표현해 효율적 학습 + 적용.
- 대표적인 modern growing recipe — 사전학습 비용 ~50% 절감하면서 quality 처음부터 학습 동등.

## 핵심 기여

- **Learnable expansion operator**: $W_{\text{large}} = M_w W_{\text{small}} M_h^\top$ 형태의 linear mapping. $M_w, M_h$ 가 학습 변수.
- **Decomposed Kronecker structure**: width 와 depth 의 expansion 을 별도 operator 로 factor — 학습 비용을 작은 matrix 의 학습으로 축소.
- **Pretrain operator on small dataset**: expansion operator 자체를 작은 dataset 의 distillation loss 로 사전학습 → 적용 시 fine-tune 없이 plug-in.
- **Multi-stage growth schedule**: 한 번에 큰 확장이 아니라 여러 단계로 — 각 stage 의 quality 추적.

## 방법 요약

- 데이터: BERT, GPT 의 표준 사전학습 corpus (BookCorpus + Wikipedia, C4 일부).
- 모델: BERT-Small → BERT-Base / BERT-Large, GPT2-Small → GPT2-Medium 등.
- 학습:
  1. Small 모델 사전학습 (수렴)
  2. LiGO operator $M_w, M_h$ 학습 — small dataset 의 reconstruction / distillation loss
  3. Operator 적용: $W_{\text{large}} = M_w W_{\text{small}} M_h^\top$
  4. Large 모델 추가 사전학습
- 핵심 수식 (width expansion of linear layer, 원논문 §3.1):

$$
W_{\text{large}} \in \mathbb{R}^{d_{\text{out}}^L \times d_{\text{in}}^L} = M_w^{\text{out}} W_{\text{small}} (M_w^{\text{in}})^\top
$$

$M_w^{\text{out}}, M_w^{\text{in}}$ 가 학습된 sparse / structured matrix.

## 실험 / 결과

- BERT-Base → BERT-Large: 처음부터 학습 대비 **40~50% step 절감** (동일 perplexity).
- bert2BERT 대비 약간 우위 + GPT 류로 일반화 가능 (bert2BERT 는 BERT-specific).
- GLUE / SQuAD downstream: 처음부터 학습 동등.
- Operator 학습 비용은 main pretrain 의 ~1% — overhead 미미.
- 재현성: 공식 PyTorch (VITA-Group) 공개.

## 한계 / 비판적 시각

- Linear operator 의 표현력 한계 — non-linear growth (예: attention head 의 functional 변화) 는 불가능.
- Operator 학습용 small dataset 의 분포가 large 모델의 target dataset 과 다르면 transfer 효과 저하.
- Width vs depth 의 expansion 비율 결정의 자동화 미해결 (사용자 hyperparameter).
- LLM-scale (7B+) 적용은 후속 MSG 가 검증.

## 본 프로젝트 시사점

- **본 프로젝트의 핵심 growing operator reference** — \"학습 중 모델 키우기\" 의 가장 일반적이고 modern 한 framework. 본 프로젝트의 `src/graphlm/training/grow.py` (가칭) 의 직접 reference.
- **차용할 아이디어**:
  - **Learnable linear expansion**: 단순 복제 (bert2BERT) 가 아니라 학습 가능 mapping → 더 나은 init quality.
  - **Decomposed Kronecker** — Memory / 계산 효율의 표준 트릭. 큰 expansion 도 작은 matrix 학습으로.
  - **Operator pretrain on small data**: main pretrain 비용을 깎으면서도 operator quality 확보.
- **채택하지 않을 부분**: 너무 복잡한 multi-stage schedule — 본 프로젝트의 초기 toy 실험에는 single-stage 로 단순화 시작.
- **후속 실험 가설**:
  - LiGO 의 학습 가능 operator vs bert2BERT 의 heuristic FPI — 작은 dataset 에서 어느 쪽이 final quality 우위인지.
  - Operator 의 sparsity / structure 가 final 모델의 표현력에 미치는 영향 — dense vs sparse / diagonal vs block-diagonal.

## 참고 / 인용

- 공식 코드: <https://github.com/VITA-Group/LiGO> (PyTorch)
- 관련 논문: [Net2Net](2016-net2net-chen.md) (origin), [bert2BERT](2022-bert2bert-chen.md) (직전 generation), [MSG](2024-msg-yao.md) (LLM 시대 확장)
- 본 프로젝트 내 인용 위치: 추후 learnable growth operator 실험 노트북에서
