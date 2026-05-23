# Papers & References

GraphLM 연구 진행에 참조하는 논문·레퍼런스의 **핵심 요약을 md 파일로 관리** 하는 디렉토리.

원문 PDF 는 가급적 외부 보관 (Zotero / Drive 등) 하고, 이 폴더에는 **요약과 본 프로젝트
시사점** 만 저장한다. 요약은 단순 번역·복사가 아니라 "왜 이 논문이 우리 작업에 관련 있는가"
를 명시한다.

## 패러다임 기반 분류

GraphLM 의 핵심 패러다임은 **training-time dynamic parameter count** (학습 중 Transformer 의 파라미터 수를 동적으로 조정) 입니다. graph 표현은 그 변동을 다루기 위한 도구. 자세한 정의는 [`CLAUDE.md`](../../CLAUDE.md#핵심-패러다임--training-time-dynamic-parameter-count) 참조.

| 카테고리 | 패러다임 정렬 | 본 프로젝트에서의 위상 |
|---|---|---|
| `graph/`, `hybrid/` (기존 8편) | **data-as-graph** | **Baseline reference 보존** — 직접 채택 대상 아님. "데이터를 graph 로" 의 비교군. |
| `computation-graph/` (현 8편) | **computation-as-graph (sparse activation 위주)** | **부분적 reference** — 8편 중 MoE/MoD/UT 6편은 \"고정 총량 + 동적 활성\" 이라 본 프로젝트 1순위가 아님. AutoFormer (NAS) / GHN-3 (arch-as-graph) 만 보조 정렬. |
| `computation-graph/` (다음 batch, 예정) | **dynamic param count (THIS)** | **1순위 큐레이션** — Growing Networks / DST / DARTS / Progressive Stacking 계열 |
| `lm/` (미사용) | LM 일반 | 필요 시 활성화 |

신규 논문 요약 추가 시 **dynamic parameter count 계열** (Growing / DST / DARTS) 을 우선한다. sparse activation (MoE 류) 와 data-as-graph 는 둘 다 reference 자료 — 직접적 baseline 가치가 명확할 때만 추가.

## 디렉토리 구조

```
docs/papers/
├── README.md         # 이 파일 — 인덱스
├── _template.md      # 신규 요약 시 복사할 표준 템플릿
├── graph/            # 순수 그래프 방법 (GCN, GAT, GraphSAGE 등)
├── lm/               # 순수 언어 모델 (BERT, GPT, T5 등)
└── hybrid/           # 그래프 + LM 결합 / multimodal
```

신규 카테고리가 필요하면 같은 깊이로 디렉토리를 추가하고 본 README 의 표를 갱신한다.

## 파일명 규칙

```
<year>-<short-id>-<first-author-lastname>.md
```

- `year`: 4자리 (출판 연도, preprint 면 arXiv 첫 등록 연도)
- `short-id`: 모델/방법 별칭 kebab-case (예: `gcn`, `bert`, `graph-bert`)
- `first-author-lastname`: 1저자 성 소문자

예시:
- `2017-gcn-kipf.md`
- `2018-bert-devlin.md`
- `2020-graph-bert-zhang.md`

## 신규 요약 추가 절차

1. `_template.md` 를 카테고리 폴더에 복사 — `cp docs/papers/_template.md docs/papers/graph/2017-gcn-kipf.md`
2. frontmatter 키 (`title, authors, year, venue, url, tags, status`) 채움
3. 본문 섹션 작성 — TL;DR 부터 차례대로
4. 본 README 의 인덱스 표에 한 줄 추가
5. `make papers-lint` 로 frontmatter 검증
6. commit prefix 는 `[DOCS]:` 사용

## Frontmatter 필수 키

| 키 | 설명 | 예시 |
|---|---|---|
| `title` | 논문 원제 | `Semi-Supervised Classification with Graph Convolutional Networks` |
| `authors` | 저자 목록 (문자열 또는 리스트) | `Kipf, T. N., & Welling, M.` |
| `year` | 출판/preprint 연도 | `2017` |
| `venue` | 학회/저널/preprint 명 | `ICLR 2017` |
| `url` | 공식 PDF 또는 arXiv URL | `https://arxiv.org/abs/1609.02907` |
| `tags` | 분류 키워드 (리스트) | `[graph, gcn, semi-supervised]` |
| `status` | 요약 작성 상태 | `draft` / `reviewed` / `archived` |

선택 키: `arxiv_id`, `doi`, `code_url`, `dataset`, `cited_in` (본 프로젝트 내 사용 위치).

## 인덱스

새 요약 추가 시 본 표에 한 줄을 더한다. 추가 컬럼은 필요 시 자유.

### Graph (data-as-graph reference)

> ⚠️ **본 프로젝트 패러다임 (computation-as-graph) 과 다른 갈래** — baseline / 비교 reference 로만 활용.

| 파일 | 연도 | 모델/방법 | 한줄 요약 |
|---|---|---|---|
| [graph/2017-gcn-kipf.md](graph/2017-gcn-kipf.md) | 2017 | GCN | Spectral conv 1차 근사 — GNN 표준 baseline |
| [graph/2017-graphsage-hamilton.md](graph/2017-graphsage-hamilton.md) | 2017 | GraphSAGE | 이웃 샘플링 기반 inductive GNN |
| [graph/2018-gat-velickovic.md](graph/2018-gat-velickovic.md) | 2018 | GAT | 노드별 attention 으로 이웃 가중치 학습 |
| [graph/2021-graphormer-ying.md](graph/2021-graphormer-ying.md) | 2021 | Graphormer | Transformer + 3종 structural bias, 분자 도메인 SOTA |

### Language Models

| 파일 | 연도 | 모델/방법 | 한줄 요약 |
|---|---|---|---|
| _(아직 없음)_ | | | |

### Hybrid (Graph + LM, data-as-graph reference)

> ⚠️ **본 프로젝트 패러다임 (computation-as-graph) 과 다른 갈래** — baseline / 비교 reference 로만 활용. 단, LM 과 graph 의 통합 패턴 측면에서 design 참고 가치 있음.

| 파일 | 연도 | 모델/방법 | 한줄 요약 |
|---|---|---|---|
| [hybrid/2020-graph-bert-zhang.md](hybrid/2020-graph-bert-zhang.md) | 2020 | Graph-BERT | 노드 subgraph 를 token 시퀀스로 변환, BERT 적용 |
| [hybrid/2021-graphformers-yang.md](hybrid/2021-graphformers-yang.md) | 2021 | GraphFormers | Transformer layer 마다 GNN aggregation nested 주입 |
| [hybrid/2023-glem-zhao.md](hybrid/2023-glem-zhao.md) | 2023 | GLEM | LM ↔ GNN 을 EM 으로 번갈아 학습, 대규모 OGB SOTA |
| [hybrid/2024-graphgpt-tang.md](hybrid/2024-graphgpt-tang.md) | 2024 | GraphGPT | Graph encoder 출력을 LLM token 으로 instruction tuning |

### Computation-graph — 현 8편 (sparse activation 위주 reference)

> ⚠️ **본 프로젝트의 1순위 (dynamic parameter count) 가 아닙니다** — 8편 중 6편 (MoE, GShard, Switch, Mixtral, MoD, Universal Transformer) 은 \"총 파라미터 고정 + 활성만 동적\" 인 sparse activation 류. AutoFormer / GHN-3 만 architecture-as-graph 측면에서 부분 정렬.
>
> 다음 batch (예정) 에서 진짜 1순위인 **Growing Networks / DST / DARTS / Progressive Stacking** 8편이 추가됩니다.

#### Sparse activation 그룹 (활성만 동적, 총량 고정)

| 파일 | 연도 | 모델/방법 | 한줄 요약 |
|---|---|---|---|
| [computation-graph/2017-moe-shazeer.md](computation-graph/2017-moe-shazeer.md) | 2017 | Sparsely-Gated MoE | Top-$k$ gating 으로 expert routing 학습 — conditional computation 의 시초 |
| [computation-graph/2019-universal-transformer-dehghani.md](computation-graph/2019-universal-transformer-dehghani.md) | 2019 | Universal Transformer | 같은 block 반복 + token 별 dynamic halting (ACT), weight-shared |
| [computation-graph/2021-gshard-lepikhin.md](computation-graph/2021-gshard-lepikhin.md) | 2021 | GShard | Top-2 gating + expert capacity + load balance — MoE scaling 표준 |
| [computation-graph/2022-switch-transformer-fedus.md](computation-graph/2022-switch-transformer-fedus.md) | 2022 | Switch Transformer | Top-1 gating 으로 MoE 단순화, 1.6T 파라미터 달성 |
| [computation-graph/2024-mod-raposo.md](computation-graph/2024-mod-raposo.md) | 2024 | Mixture of Depths | Expert-choice router 가 layer 별 token 통과 여부 결정 — layer routing |
| [computation-graph/2024-mixtral-jiang.md](computation-graph/2024-mixtral-jiang.md) | 2024 | Mixtral 8×7B | 8 expert top-2 MoE LLM, open-weight, 현대 MoE production reference |

#### Architecture-as-graph 그룹 (부분 정렬)

| 파일 | 연도 | 모델/방법 | 한줄 요약 |
|---|---|---|---|
| [computation-graph/2021-autoformer-chen.md](computation-graph/2021-autoformer-chen.md) | 2021 | AutoFormer | NAS for Transformer, weight entanglement 로 supernet 동시 학습 — 학습 후 fixed |
| [computation-graph/2023-ghn3-knyazev.md](computation-graph/2023-ghn3-knyazev.md) | 2023 | GHN-3 | NN architecture 를 DAG 으로, hypernetwork 가 weight 예측 — 학습 중 변동 X |

### Dynamic param count (예정, 다음 PR — 본 프로젝트 진짜 1순위)

다음 batch 에서 \`docs/papers/computation-graph/\` 에 추가될 8편:
- Net2Net (2016), SET (2018), Progressive Stacking BERT (2019), DARTS (2019), RigL (2020), bert2BERT (2022), LiGO (2023), MSG (2024)

## 작성 원칙

- **요약은 자기 언어로** — 논문 abstract 복붙 금지. 본인이 이해한 표현으로 재기술
- **"본 프로젝트 시사점" 섹션 필수** — 단순 요약이 아니라 GraphLM 에 어떻게 적용 가능한지
- **출처 명확화** — **비자명한 구체 사실 / 정확한 수식** 을 인용할 때 페이지/섹션 명시 (예: `(§3.2)`, `(Table 1)`). high-level overview 요약 (각 논문의 abstract / main result 수준) 에는 적용 강제 X — 후속 deep-dive 시 가능.
- **수식은 LaTeX** — `$...$` (inline) 또는 `$$...$$` (block) 사용. 한국어 문맥과 섞일 때
  공백 1칸씩 유지하여 가독성 확보
- **status 갱신** — 초안 작성 시 `draft` → 본인 1차 리뷰 후 `reviewed` → 후속 작업으로
  더 이상 활성 참조 안 하면 `archived`

상세 작성 컨벤션은 [`_template.md`](_template.md) 의 주석 참조.
