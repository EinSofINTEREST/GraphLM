# Notion sync — Papers DB

GraphLM 의 `docs/papers/**/*.md` 를 Notion 의 **Papers** data source 로 동기화합니다.

## Notion 위치

- Workspace: `EinSof : INTEREST` → `Projects` DB → **GraphLM** page
- Sub-page: **참고문헌 (References)** → **Papers** DB
- URL: https://www.notion.so/36ae8b70b7aa81f79689eb53b7f57cfa
- Data source ID: `a71d6289-d13a-4d4a-a66d-ff52df038939`

## 스크립트

`scripts/notion_papers_sync.py` — IssueTracker 의 `notion_pr_sync.py` 를 paper 용으로 변형.

### 사전 조건

`NOTION_API_TOKEN` 환경변수 — `.env` 의 `NOTION_API_TOKEN=ntn_...` 사용.

```bash
export NOTION_API_TOKEN=$(grep '^NOTION_API_TOKEN=' .env | cut -d= -f2-)
```

### 모드

#### `file` — 단일 paper sync

새 paper 추가 / 기존 paper 갱신 시:

```bash
python3 scripts/notion_papers_sync.py --mode file \
  --path docs/papers/computation-graph/2024-msg-yao.md
```

⚠️ 현재 file 모드는 **항상 새 row 생성** (idempotent X). 동일 paper 의 row 가 중복되지 않게 하려면, 갱신 전에 Notion 에서 기존 row 를 archive 하거나 backfill 사용.

#### `backfill` — 전체 재구성

```bash
python3 scripts/notion_papers_sync.py --mode backfill
```

기존 모든 row archive → `docs/papers/` 의 모든 md 를 새로 생성.

#### `backfill --skip-existing` — 누락 보완

```bash
python3 scripts/notion_papers_sync.py --mode backfill --skip-existing
```

`별칭` (short id) 기준으로 이미 존재하는 row 는 skip — backfill 중단 후 재시작 시 idempotent resume.

### 동작

1. `docs/papers/**/*.md` 에서 frontmatter + body 추출 (`README.md` / `_template.md` 제외)
2. frontmatter → Notion properties 매핑:
   - `제목` ← `title`
   - `별칭` ← 파일명에서 추출 (`2017-gcn-kipf.md` → `GCN`)
   - `연도` ← `year`
   - `venue` ← `venue` (VENUE_MAP 으로 select option 매핑)
   - `카테고리` ← 디렉토리 (graph / hybrid / computation-graph)
   - `갈래` ← `tags` (TAG_TO_BRANCH 로 multi_select 매핑)
   - `paradigm 정렬` ← 카테고리 + tags 로 자동 결정
   - `status` ← `status`
   - `URL` ← `url`
   - `코드` ← `code_url`
   - `Local 경로` ← repo 내 상대 경로
3. body markdown → Notion blocks (`md_to_blocks`)
4. 각 row 의 page content 마지막에 GitHub blob URL 링크 추가

### 매핑 사용자화

신규 venue / tag 추가 시 `scripts/notion_papers_sync.py` 의 다음 dict 갱신:
- `VENUE_MAP`
- `TAG_TO_BRANCH`
- `CATEGORY_MAP`

Notion DB 의 select option 도 함께 추가 필요 (MCP 또는 Notion UI).

## 향후 자동화 (TODO)

현재는 수동 실행. GitHub Actions workflow 로 자동 sync 가능 (IssueTracker 의 `.github/workflows/notion-pr-sync.yml` 패턴):
- `push` to `main` + `docs/papers/**` paths filter → `--mode file` 호출 (변경된 paper 만)
- `workflow_dispatch` → `--mode backfill` 전체 재구성

본 PR 의 scope 밖 — 별도 issue 로 진행 가능.

## 관련 파일

- `scripts/notion_papers_sync.py` — sync 스크립트
- `docs/papers/` — paper md 원본
- `.env` — `NOTION_API_TOKEN` (gitignore)
