---
title: "Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science"
authors: "Mocanu, D. C., Mocanu, E., Stone, P., Nguyen, P. H., Gibescu, M., & Liotta, A."
year: 2018
venue: "Nature Communications 2018"
url: "https://www.nature.com/articles/s41467-018-04316-3"
arxiv_id: "1707.04780"
code_url: "https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks"
tags: ["computation-graph", "dynamic-param", "set", "dst", "sparse", "evolutionary", "prune-grow", "foundation"]
status: "draft"
cited_in: []
---

# SET — Sparse Evolutionary Training (DST 의 시초)

## TL;DR (3줄)

- 학습 초기부터 **sparse mask 로 시작** 한 뒤, 매 epoch 마다 작은 weight 를 **prune** + 같은 수의 random connection 을 **regrow** → connection 집합이 학습 중 진화.
- 동일 sparsity 에서 dense 학습 후 prune 보다 더 높은 quality + 학습 비용 ~10x 절감.
- 본 프로젝트 패러다임 **\"학습 중 파라미터 수 동적 조정\"** 의 connection-level 시초. \"활성 파라미터 집합이 매 epoch 변동\" 의 첫 증명.

## 핵심 기여

- **Sparse-to-sparse training**: dense 로 시작해 prune 하는 lottery ticket 류와 달리, **처음부터 sparse** 로 학습 → 메모리 / FLOP 절감.
- **Prune-and-regrow per epoch**: 매 epoch 끝에 magnitude 가 작은 connection 의 일정 fraction $\zeta$ 제거 + 동일 수의 random new connection 추가.
- **Erdős–Rényi initialization**: sparse mask 의 초기 분포를 network science 의 random graph 모델로 — 균등 분포보다 quality 우위.
- **Generic framework**: MLP, RBM, CNN 에 모두 적용. 후속 RigL (gradient-based regrowth) 의 기반.

## 방법 요약

- 데이터: MNIST, CIFAR-10, COIL-20 (image), Sensorless drive diagnosis, MAGIC, BACH (UCI).
- 모델: sparse MLP / RBM / CNN. sparsity $\epsilon \in \{0.05, 0.1, 0.2\}$ (전체 connection 중 활성 비율).
- 학습:
  1. Erdős–Rényi random sparse mask init
  2. Standard SGD epoch
  3. Epoch 끝: 작은 weight $\zeta$ 비율 제거 + 동일 수 random regrow
  4. 반복
- 핵심 흐름:

매 epoch $t$:
$$
\mathcal{M}^{(t+1)} = (\mathcal{M}^{(t)} \setminus \text{Prune}_\zeta(\mathcal{M}^{(t)})) \cup \text{RandomGrow}(\zeta \cdot |\mathcal{M}^{(t)}|)
$$

여기서 $\mathcal{M}^{(t)}$ 는 epoch $t$ 의 활성 connection 집합.

## 실험 / 결과

- MNIST MLP: sparse 96.5% (5% connection) vs dense 96.7% → 동급 quality, 20x 파라미터 감소.
- CIFAR-10 CNN: 비슷한 패턴.
- 학습 시간: sparse-from-start 가 dense + prune 류 대비 큰 폭 절감.
- 재현성: 공식 NumPy + Theano 구현 공개 (오래된 의존성). 후속 RigL 등의 PyTorch 재구현이 표준 reference.

## 한계 / 비판적 시각

- **Random regrowth** 가 비효율 — 어떤 connection 이 학습에 유용할지에 대한 신호 없음. RigL 이 gradient magnitude 로 개선.
- Sparsity ratio 가 작아지면 (5% 이하) quality 급격 저하 — extreme sparse 영역에서 한계.
- 평가가 vision / table data 위주. Transformer / LM 에 직접 적용은 후속 작업 (RigL, Top-KAST 등).
- 학습 시간 측정의 fair comparison 어려움 — sparse op 의 GPU 가속이 dense 보다 느릴 수 있음 (구현 의존).

## 본 프로젝트 시사점

- **Connection-level dynamic param 의 reference foundation** — 본 프로젝트가 \"connection mask 가 학습 중 진화하는\" 류 모델을 다룬다면 SET 가 최소 baseline.
- **차용할 아이디어**:
  - **Prune-and-regrow per epoch loop** — 본 프로젝트의 PyTorch 구현 시 `epoch_callback` 으로 mask 갱신하는 패턴.
  - **Erdős–Rényi init** — uniform 보다 더 다양한 connection 분포로 시작.
  - **Layer 별 sparsity 분리** — 모든 layer 균등이 아니라 layer 별 다른 sparsity 가능.
- **채택하지 않을 부분**: random regrowth 의 비효율 — RigL 의 gradient-based 가 직접 reference.
- **후속 실험 가설**:
  - SET 의 sparse-from-start vs RigL 의 magnitude-pruning + gradient-regrowth 비교 — small Transformer 에서 어느 쪽이 학습 안정성과 quality 우위인지.
  - $\zeta$ (prune fraction per epoch) 와 sparsity ratio $\epsilon$ 의 trade-off 측정.

## 참고 / 인용

- 공식 코드: <https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks> (Python + Theano)
- 관련 논문: [RigL](2020-rigl-evci.md) (현대 DST), [Net2Net](2016-net2net-chen.md) (다른 방향의 dynamic param)
- 본 프로젝트 내 인용 위치: 추후 DST 실험 노트북에서
