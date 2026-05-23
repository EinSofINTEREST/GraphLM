---
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
authors: "Fedus, W., Zoph, B., & Shazeer, N."
year: 2022
venue: "JMLR 2022"
url: "https://arxiv.org/abs/2101.03961"
arxiv_id: "2101.03961"
code_url: "https://github.com/tensorflow/mesh"
tags: ["computation-graph", "moe", "switch-transformer", "top-1-gating", "scaling", "sparsity"]
status: "draft"
cited_in: []
---

# Switch Transformer — Top-1 Gating 의 단순함 + 효율

## TL;DR (3줄)

- GShard 의 top-2 gating 을 **top-1 (단 하나의 expert)** 로 단순화 — communication 비용 절반, 구현 단순, 학습 안정성 향상.
- 1.6T 파라미터 모델을 T5-XXL (11B dense) 대비 동일 compute 로 학습 → 4-7x speedup at fixed quality.
- \"Simpler is better\" — MoE 의 design space 를 줄이면서도 효과는 유지/향상 시킨 mile stone. 본 프로젝트의 **default MoE 후보**.

## 핵심 기여

- **Top-1 gating with simplification**: 각 token 이 **단 하나의 expert** 로만 routing. capacity factor (1.0~1.5) 로 capacity 제어.
- **Differentiable routing weight as scaling**: 선택된 expert 의 output 에 gate 의 soft prob 을 곱함 → gradient 가 gate 로 흐름.
- **Auxiliary load balancing loss 단순화**: $\alpha \cdot N \cdot \sum_i f_i \cdot P_i$ — fraction $f_i$ × probability $P_i$. GShard 보다 단순.
- **Selective precision (bfloat16/float32 hybrid)**: routing 결정은 float32, expert 계산은 bfloat16 → 학습 안정성 + 메모리 효율.

## 방법 요약

- 데이터: C4 (사전학습), GLUE / SuperGLUE / SQuAD (다운스트림), mC4 (multilingual).
- 모델: T5 기반 + FFN 을 Switch layer (top-1 MoE) 로 교체. expert 수 $N \in \{8, 32, 128, 512, 2048\}$, 최대 1.6T parameters.
- 학습: Adafactor, T5 표준 setup. TPU v3 pod.
- 핵심 수식:

$$
y = G(x)_{i^*} \cdot E_{i^*}(x), \quad i^* = \arg\max_i G(x)_i, \quad G(x) = \text{softmax}(xW_g)
$$

단일 expert 선택, soft gate weight 곱하기.

## 실험 / 결과

- C4 사전학습 perplexity: Switch-Base (7B) 가 T5-Base (220M) 대비 7x 빠른 step-당 학습.
- Switch-XXL (395B) 가 T5-XXL (11B) 대비 4x training speedup at same quality.
- 다운스트림 task (SuperGLUE, XSum, ANLI 등) 일관 우위.
- 1.6T 파라미터 모델 학습 — 당시 최대.
- 재현성: 공식 TensorFlow (mesh) + 후속 PyTorch 포트 (DeepSpeed-MoE, FairScale) 다수.

## 한계 / 비판적 시각

- Top-1 은 표현력 측면에서 top-2 보다 약함 — 특정 task 에서 GShard 가 우위 보고도 있음.
- Expert 수 늘릴수록 memory 가 폭증 (각 expert 는 dense FFN) — small GPU 에선 비현실.
- Routing 의 stochasticity 가 작아 expert collapse 위험 — 보조 loss 의 hyperparameter 여전히 중요.
- Multilingual / multi-domain 에서 expert 가 \"언어/도메인 specialization\" 으로 수렴하는지 해석성 부족.

## 본 프로젝트 시사점

- **본 프로젝트의 default MoE 구현 후보 1순위** — top-1 의 단순함이 PyTorch 구현 비용을 크게 낮춤. 작은 GPU 환경에서도 toy version 시작 가능.
- **차용할 아이디어**:
  - **Top-1 + soft gate weight** 의 minimal recipe — 본 프로젝트의 첫 MoE 모델 (`src/graphlm/models/switch_moe.py`) 의 직접 reference.
  - **단순화된 load balancing loss** — GShard 의 복잡한 수식보다 구현 친화적.
  - **Selective precision** — fp16/bf16 학습 시 routing 만 fp32 로 — Mixtral 도 동일 패턴.
- **채택하지 않을 부분**: TPU SPMD 분산 — 본 프로젝트는 PyTorch FSDP 또는 단일 GPU.
- **후속 실험 가설**: 동일 active parameter 수 대비 (Switch-top1) vs (GShard-top2) 의 perplexity 비교. expert 수 8개 정도 작은 setup 에서 routing 결정의 stability 와 expert collapse rate.

## 참고 / 인용

- 공식 코드: <https://github.com/tensorflow/mesh> (TensorFlow), DeepSpeed-MoE / FairScale (PyTorch)
- 관련 논문: [Sparsely-Gated MoE](2017-moe-shazeer.md) (조상), [GShard](2021-gshard-lepikhin.md) (top-2 대비군), [Mixtral](2024-mixtral-jiang.md) (현대 LLM 적용)
- 본 프로젝트 내 인용 위치: 추후 첫 MoE 모델 구현 / 노트북에서
