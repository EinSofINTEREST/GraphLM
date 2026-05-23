---
title: "Universal Transformers"
authors: "Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł."
year: 2019
venue: "ICLR 2019"
url: "https://arxiv.org/abs/1807.03819"
arxiv_id: "1807.03819"
code_url: "https://github.com/tensorflow/tensor2tensor"
tags: ["computation-graph", "universal-transformer", "adaptive-depth", "halting", "recurrent", "act"]
status: "draft"
cited_in: []
---

# Universal Transformer — Adaptive Computation Depth

## TL;DR (3줄)

- 같은 Transformer block 을 **recurrent 하게 반복** 적용하면서, **각 position 마다 독립적으로 \"몇 번 반복할지\"** 를 dynamic halting (ACT) 으로 결정.
- 표준 Transformer 의 fixed depth 한계를 깨고, 입력 복잡도에 따라 token 별로 다른 computation 깊이 사용 → algorithmic / reasoning task 에서 큰 폭 개선.
- 본 프로젝트 패러다임의 **\"각 token 이 layer graph 위에서 자기 path 를 결정\"** 의 첫 사례.

## 핵심 기여

- **Weight-shared depth**: $N$ 번 반복되는 layer 가 모두 같은 파라미터 → 모델 크기는 단일 layer, computation 만 token 별로 가변.
- **Adaptive Computation Time (ACT) for token halting**: 각 position 마다 누적 halting probability 가 임계값 도달 시 계산 중단 → token 별 동적 depth.
- **Universality (이론적 주장)**: Turing-complete 한 RNN 의 한 형태로 볼 수 있음 — 알고리즘 학습 능력 강화.

## 방법 요약

- 데이터: bAbI QA, algorithmic tasks (copy, reverse, addition), WMT'14 En-De, LAMBADA.
- 모델: standard Transformer encoder 의 block 을 $N$ 번 (또는 ACT 가 정한 만큼) 재적용. position 마다 별도 halting unit.
- 학습: standard cross-entropy + ACT 의 ponder cost (computation 낭비 페널티).
- 핵심 흐름:

$$
h_v^{(t+1)} = \text{TransformerBlock}(h_v^{(t)} + p_v^{(t+1)}), \quad p_v^{(t)} = \text{PositionalEncoding}(v, t)
$$

각 position $v$ 마다 halting $h_v$ 가 threshold 도달 시 update 중단.

## 실험 / 결과

- bAbI: 20개 task 중 평균 fail task 0.23 vs Transformer baseline 1.30.
- Algorithmic (copy/reverse/addition): generalization length 에서 Transformer 대비 큰 폭.
- WMT'14 En-De BLEU 28.9 vs Transformer-base 27.3.
- 재현성: tensor2tensor 의 공식 구현. 후속 PyTorch 포트도 존재.

## 한계 / 비판적 시각

- **Weight sharing 의 capacity 손실** — 동일 파라미터를 반복하니 표현력은 dense Transformer 만 못 함. 후속 작업이 sparse MoE 와 결합 (Sparse Universal Transformer, 2023).
- ACT 의 학습이 불안정 — halting threshold 와 ponder cost 의 hyperparameter 민감.
- Reasoning task 위주 강세, 일반 NLP 에서 효과 마진 작음.
- 추론 시 동적 depth 가 batch 추론에 비효율 (token 마다 다른 step 수).

## 본 프로젝트 시사점

- **\"Token 별 동적 computation path\" 의 초기 증명** — 본 프로젝트의 핵심 가설 (token 이 layer graph 위에서 routing) 의 직계 조상. Mixture of Depths 가 이 아이디어를 sparse 화한 후속작.
- **차용할 아이디어**:
  - **Halting unit per position** — token 별 layer skip 결정을 위한 sigmoid head 의 reference design.
  - **Ponder cost** — computation 비용 페널티를 loss 에 명시 → routing 의 efficiency 학습.
- **채택하지 않을 부분**: weight sharing 의 capacity 손실 — 본 프로젝트는 layer 마다 다른 expert 두는 MoE 와 결합 가능성 우선.
- **후속 실험 가설**: Universal Transformer 의 token-level halting + Mixture of Depths 의 layer-level skip 의 조합 — 어느 쪽이 작은 dataset 에서 더 효과적인지 ablation.

## 참고 / 인용

- 공식 코드: tensor2tensor `transformer_universal` (TensorFlow)
- 관련 논문: [Mixture of Depths](2024-mod-raposo.md) (sparse 후속), [Sparsely-Gated MoE](2017-moe-shazeer.md) (다른 routing 방향), Sparse Universal Transformer (Tan et al., NeurIPS 2023)
- 본 프로젝트 내 인용 위치: 추후 adaptive-depth 실험 노트북에서
