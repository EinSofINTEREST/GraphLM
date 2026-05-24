---
title: "AtomNAS: Fine-Grained End-to-End Neural Architecture Search"
authors: "Mei, J., Li, Y., Lian, X., Jin, X., Yang, L., Yuille, A., & Yang, J."
year: 2020
venue: "ICLR 2020"
url: "https://arxiv.org/abs/1912.09640"
arxiv_id: "1912.09640"
tags: ["computation-graph", "nas", "fine-grained", "atomic-block", "function-level", "end-to-end"]
status: "draft"
cited_in: []
---

# AtomNAS — Atomic Block 단위의 Fine-Grained NAS

## TL;DR (3줄)

- 기존 NAS 가 cell / block 단위로 search 하는 반면 **AtomNAS 는 atomic block** (channel + kernel size 단위의 최소 단위) 으로 더 fine-grained search.
- **End-to-end** — search 와 학습을 같은 process 에서 수행 (기존 search → retrain 의 2-stage 회피).
- **Dynamic network shrinkage** — 학습 중 영향이 없는 atomic block 을 on-the-fly prune.

## 핵심 기여

- **Atomic block 정의** — 한 layer 안의 individual channel + kernel choice (e.g. 3×3 / 5×5 / 7×7 conv 의 mixture) 가 minimal search unit. 기존 NAS 의 layer block 보다 훨씬 작음.
- **Supernet 의 atomic 다양성** — 같은 layer 안에 여러 atomic block 이 parallel 로 공존, 학습이 가장 유용한 것을 select.
- **On-the-fly shrinkage** — output 에 미세 영향만 주는 atomic block 을 학습 중 자동 prune. search + train 이 통합.

## 방법 요약

- 데이터: ImageNet (image classification).
- 모델: CNN supernet (다양한 channel × kernel 조합의 atomic block).
- 학습:
  - Supernet 학습 (모든 atomic block 활성)
  - 각 atomic block 의 magnitude (output norm) 모니터링
  - 임계 이하 prune (BN scale γ 가 0 근처가 되는 atomic block)
- 핵심: BN 의 γ 가 architecture mask 역할 — γ=0 인 atomic block 은 "off".

## 실험 / 결과

- ImageNet: 동일 FLOPs 대비 SOTA accuracy.
- 학습 시간: 기존 NAS (e.g. NASNet, DARTS) 대비 search overhead 거의 0 (end-to-end).
- 재현성: 공식 PyTorch 구현 공개.

## 한계 / 비판적 시각

- **CNN 위주** — Transformer 적용 X (channel 개념의 직접 mapping 어려움).
- BN γ 의존 — Pre-LN Transformer (γ 다른 의미) 에는 직접 적용 어려움.
- **정적 architecture 결과** — search 끝나면 architecture 고정. 사용자 컨셉 (지속적 동적 변화) 와 다름.

## 본 프로젝트 시사점

> **Fine-grained 노드 단위 search 의 가장 대표 선례** — 본 프로젝트의 "function-level 노드" 컨셉의 NAS 차원 reference.

- **적용 가능 부분**:
  - Atomic block 의 정의 (한 layer 안의 individual choice) 를 Transformer 의 sub-component (qkv / fc1 / fc2 / activation function) 단위로 일반화
  - BN γ 의 prune 신호 → Phase 1 의 α 분포 분석 (#4) 의 정량 패턴과 유사 — α ≈ 0 인 block 을 prune 신호로 활용 가능
- **차용할 아이디어**:
  - **On-the-fly prune** — Phase 1 에는 prune 없음. dead block 자동 제거에 직접 활용 가능
  - **Magnitude-based mask** (BN γ) — 본 프로젝트의 α 가 본질적으로 같은 역할. AtomNAS 의 γ schedule 을 α 에 적용 가능
- **채택하지 않을 부분**:
  - CNN-specific 의 channel/kernel mixture
  - 정적 search-then-train 의 2 phase 분리 (본 프로젝트는 학습 중 dynamic 우선)
- **후속 실험 가설**:
  - Phase 1 의 α ≈ 0 인 block 을 prune 하면 final loss 가 baseline 과 같은가 (즉 정말 dead 여서 제거해도 무관한가)?
  - AtomNAS 의 supernet → AtomNAS-like Transformer 의 atomic = qkv vs fc1 vs activation. parallel atomic 의 selection 학습이 가능한가?

## 참고 / 인용

- 공식 코드: <https://github.com/meijieru/AtomNAS> (PyTorch)
- 관련 논문: [DARTS](2019-darts-liu.md) (가장 직접 baseline), [FGNAS](https://arxiv.org/pdf/1911.07478) (channel-level fine-grained), [Once-for-All](2019-once-for-all-cai.md) (supernet)
- 본 프로젝트 내 인용 위치: function-level 노드 컨셉의 NAS 차원 reference
