# Notebooks

GraphLM 실험 노트북 디렉토리.

## 명명 규칙

```
<번호 prefix>/<번호>-<주제 kebab>.ipynb
```

| Prefix | 단계 | 예시 |
|---|---|---|
| `00-exploration/` | 데이터 탐색, EDA | `00-cora-dataset-stats.ipynb` |
| `10-experiments/` | 가설 검증 학습 실험 | `10-gcn-baseline.ipynb`, `11-attention-vs-gcn.ipynb` |
| `20-analysis/` | 결과 분석, 시각화 | `20-attention-head-vis.ipynb` |

## 노트북 작성 규칙

- 상단 markdown 셀에 가설 / 데이터 / 시드 / 연관 이슈 명시
- 로직은 `src/graphlm/` 모듈에서 import — 셀 안에서 함수/클래스 정의 금지
- 하이퍼파라미터는 단일 config 셀에 모음 (다른 셀에서 하드코딩 금지)
- 커밋 전 `make nb-clean` 으로 output cell 제거

상세는 [.claude/rules/01-architecture.md](../.claude/rules/01-architecture.md) 의
"코드 / 노트북 분리 원칙" 참조.

## 예시 노트북 헤더

```markdown
# 10-gcn-baseline

- 가설: GCN baseline 이 Cora 데이터셋에서 80% 이상 accuracy 달성
- 데이터: Cora (planetoid)
- 시드: 42
- 작성일: 2026-05-14
- 연관 이슈: #15
```
