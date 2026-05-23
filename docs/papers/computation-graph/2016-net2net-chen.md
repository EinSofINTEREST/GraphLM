---
title: "Net2Net: Accelerating Learning via Knowledge Transfer"
authors: "Chen, T., Goodfellow, I., & Shlens, J."
year: 2016
venue: "ICLR 2016"
url: "https://arxiv.org/abs/1511.05641"
arxiv_id: "1511.05641"
code_url: "https://github.com/soumith/net2net.torch"
tags: ["computation-graph", "dynamic-param", "net2net", "growing", "function-preserving", "knowledge-transfer", "foundation"]
status: "draft"
cited_in: []
---

# Net2Net — Function-Preserving Network Growth 의 시초

## TL;DR (3줄)

- 작은 \"teacher\" 신경망의 weight 를 더 큰 \"student\" 신경망으로 **function-preserving 하게 mapping** 하는 두 가지 operator: **Net2WiderNet** (layer 폭 확장) + **Net2DeeperNet** (layer 깊이 추가).
- 확장 직후 student 의 output = teacher 의 output (정확히 동일) → 처음부터 학습 대비 훨씬 빠른 수렴, 사전 학습된 capacity 그대로 보존.
- 본 프로젝트 패러다임 **\"학습 중 파라미터 수 동적 조정\"** 의 foundation 문헌. 모든 후속 growing methods (bert2BERT, LiGO, MSG) 가 이 function-preserving 아이디어의 변주.

## 핵심 기여

- **Net2WiderNet**: layer $i$ 의 width 를 $n \to m$ (where $m > n$) 으로 확장 — 기존 $n$ neuron 의 weight 를 복제하고 다음 layer 의 incoming weight 를 그 복제 수로 나눠 출력 보존.
- **Net2DeeperNet**: 새 layer 를 identity initialization (단순 linear/ReLU 의 경우) 으로 삽입 — output 변경 없이 깊이만 추가.
- **Function preservation 보장**: 확장 직후 모든 input 에 대해 $f_{\text{student}}(x) = f_{\text{teacher}}(x)$. 후속 학습은 이 \"warm start\" 에서 진행.
- **Knowledge transfer 의 단순화**: distillation / quantization 등 복잡한 transfer 대비 weight 복사 + scaling 만으로 동등한 효과.

## 방법 요약

- 데이터: ImageNet (Inception-BN baseline).
- 모델: convolutional networks (Inception, AlexNet 변형).
- 학습:
  1. Teacher 학습 (수렴까지)
  2. Net2WiderNet / Net2DeeperNet 으로 확장
  3. Student 를 teacher 의 weight 로 init → 동일 data 로 추가 학습
- 핵심 수식 (Net2WiderNet, 원논문 §2.1 Eq. 1):

$$
\text{new layer}_{i+1}\text{ weight}: U^{(i+1)}_{k, j} =
\begin{cases}
W^{(i+1)}_{k, j} & j \le n \\
\frac{1}{|\{l : g(l) = g(j)\}|} W^{(i+1)}_{k, g(j)} & j > n
\end{cases}
$$

여기서 $g(j)$ 는 새 neuron $j$ 가 어느 기존 neuron 을 복제했는지의 mapping.

## 실험 / 결과

- ImageNet Inception-BN: teacher 학습 + Net2DeeperNet 확장 학습 = 직접 큰 모델 학습 대비 **수렴 속도 ~2x**, 최종 accuracy 동등 또는 약간 우위.
- Net2WiderNet 도 비슷한 효율 — 폭만 늘리는 단순 reinit 대비 큰 폭.
- 재현성: 공식 Torch 구현 (Soumith Chintala 의 포트) 공개. PyTorch 후속 재구현 다수.

## 한계 / 비판적 시각

- **Function preservation 가정의 좁음** — ReLU + Batch Norm 의 특정 조합에서만 깔끔. modern Transformer 의 LayerNorm + residual + multi-head attention 에는 직접 적용 어려움 (후속 bert2BERT, LiGO 가 확장).
- 확장 시점 / 확장 폭의 schedule 이 hyperparameter — 자동 결정 미해결.
- 학습된 weight 의 정확한 복제 — 표현 공간의 redundancy 가 그대로 student 로 전이 → 일부 weight 가 학습 후에도 거의 동일.
- 신경망 외 generic ML model 에는 적용 불가.

## 본 프로젝트 시사점

- **모든 growing methods 의 reference foundation** — 본 프로젝트가 \"학습 중 파라미터 수를 늘리는\" 모듈을 구현할 때 Net2Net 의 function-preserving 원칙이 design contract.
- **차용할 아이디어**:
  - **Function preservation 보장 후 확장** 의 두 단계 분리 — \"warm start 직후 동일 output\" 검증을 unit test 로 만들면 안전성 보장 가능.
  - **Identity initialization** (Net2DeeperNet) — Transformer 의 새 layer 를 zero-init residual scale 로 추가하는 패턴의 직접 reference.
- **채택하지 않을 부분**: CNN-specific 의 weight 복제 + 1/count scaling 은 그대로 적용 안 됨 — LiGO 의 학습 가능 linear operator 가 일반화.
- **후속 실험 가설**:
  - 작은 toy Transformer 에서 Net2WiderNet (hidden dim 확장) + Net2DeeperNet (layer 추가) 의 function preservation 을 직접 검증.
  - 두 operator 의 \"확장 횟수 vs 최종 quality\" trade-off 측정 — 한 번에 큰 폭 vs 여러 번 작은 폭.

## 참고 / 인용

- 공식 코드: <https://github.com/soumith/net2net.torch> (Torch)
- 관련 논문: [bert2BERT](2022-bert2bert-chen.md) (BERT 도메인 확장), [LiGO](2023-ligo-wang.md) (학습 가능 일반화), [MSG](2024-msg-yao.md) (LLM 시대 적용)
- 본 프로젝트 내 인용 위치: 추후 growing Transformer 첫 노트북에서
