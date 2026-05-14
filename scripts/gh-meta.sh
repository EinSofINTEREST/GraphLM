#!/usr/bin/env bash
# gh-meta.sh — 이슈/PR 의 title prefix 를 읽어 Label · Issue Type 을 자동 부여합니다.
#
# 사용법:
#   scripts/gh-meta.sh issue <NUMBER>   # 이슈 label + type 부여
#   scripts/gh-meta.sh pr    <NUMBER>   # PR label 부여 (닫는 이슈의 label 와 동기화)
#
# 환경변수:
#   GH_REPO                       — 저장소 (예: myorg/GraphLM). 미지정 시 gh repo view 자동 추출
#   GRAPHLM_FEATURE_TYPE_ID       — Issue Type GraphQL ID (Feature). 미지정 시 type 부여 skip
#   GRAPHLM_BUG_TYPE_ID           — Issue Type GraphQL ID (Bug)
#   GRAPHLM_TASK_TYPE_ID          — Issue Type GraphQL ID (Task)
#
#   Type ID 조회 쿼리:
#     gh api graphql -f query='{ repository(owner:"OWNER",name:"REPO"){
#       issueTypes(first:10){ nodes{ id name } } } }'
#
# prefix → label / type 매핑 (07-workflow.md 규약 6):
#   이슈 title:  [FEATURE] / [FIX] / [HOTFIX] / [REFACTOR] / [CHORE] / [DOCS]
#   PR title:   [FEAT#N]  / [FIX#N] 등 — prefix 부분([A-Z]+)만 추출해 매핑

set -euo pipefail

# REPO 는 환경변수 GH_REPO 로 override 가능. 기본값은 현재 git remote 의 OWNER/REPO 추출.
# 예: GH_REPO=myorg/GraphLM scripts/gh-meta.sh issue 15
REPO="${GH_REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")}"

if [[ -z "$REPO" ]]; then
  echo "ERROR: 저장소를 식별할 수 없습니다. GH_REPO 환경변수로 명시하거나 gh repo view 가 동작하는 디렉토리에서 실행하세요." >&2
  exit 1
fi

# Issue Type ID — 본 repo 의 GraphQL ID. 신규 repo 라면 다음 쿼리로 조회 후 갱신:
#   gh api graphql -f query='{ repository(owner:"OWNER",name:"REPO"){ issueTypes(first:10){ nodes{ id name } } } }'
FEATURE_TYPE_ID="${GRAPHLM_FEATURE_TYPE_ID:-}"
BUG_TYPE_ID="${GRAPHLM_BUG_TYPE_ID:-}"
TASK_TYPE_ID="${GRAPHLM_TASK_TYPE_ID:-}"

usage() {
  echo "Usage: $0 issue <NUMBER> | pr <NUMBER>" >&2
  exit 1
}

# prefix → (label, type_id) 반환
# 이슈 [FEATURE] 와 PR [FEAT#N] 양쪽을 모두 처리 (gemini 피드백 반영 — PR #239)
resolve_label_and_type() {
  local title="$1"
  local prefix
  # [FEATURE], [FEAT#123], [DOCS] 등에서 대괄호 안의 알파벳 부분만 추출
  # POSIX 호환: []#] 는 ] 를 문자 클래스 첫 자리에 놓아 리터럴로 인식 — ] 또는 # 필수 (PR #244 Copilot)
  prefix=$(echo "$title" | sed -En 's/^\[([A-Z]+)[]#].*/\1/p' || true)

  case "$prefix" in
    FEATURE|FEAT)    echo "enhancement ${FEATURE_TYPE_ID}" ;;
    BUG|FIX)         echo "bug ${BUG_TYPE_ID}" ;;
    HOTFIX)          echo "bug+hotfix ${BUG_TYPE_ID}" ;;
    REFACTOR|REFAC)  echo "refactor ${TASK_TYPE_ID}" ;;
    CHORE)           echo "chore ${TASK_TYPE_ID}" ;;
    DOCS)            echo "documentation ${TASK_TYPE_ID}" ;;
    *)
      echo "ERROR: unrecognized prefix '$prefix' in title: $title" >&2
      exit 1
      ;;
  esac
}

apply_issue_label_and_type() {
  local number="$1"
  local title
  title=$(gh issue view "$number" --repo "$REPO" --json title --jq .title)
  echo "Issue #$number: $title"

  local meta
  meta=$(resolve_label_and_type "$title")
  local label="${meta%% *}"
  local type_id="${meta##* }"

  # label 적용 — +를 ,로 변환해 한 번에 전달
  gh issue edit "$number" --repo "$REPO" --add-label "${label//+/,}"
  echo "  label: ${label//+/,}"

  # Issue Type 적용 — TYPE_ID 환경변수가 비어있으면 skip (GraphLM 신규 repo 에선 ID 설정 필요)
  if [[ -z "$type_id" ]]; then
    echo "  (skip type: GRAPHLM_*_TYPE_ID 환경변수 미설정 — README 참조)"
    return 0
  fi
  local issue_id
  issue_id=$(gh issue view "$number" --repo "$REPO" --json id --jq .id)
  gh api graphql -f query='
mutation($issueId: ID!, $issueTypeId: ID!) {
  updateIssueIssueType(input: {issueId: $issueId, issueTypeId: $issueTypeId}) {
    issue { number issueType { name } }
  }
}' -f issueId="$issue_id" -f issueTypeId="$type_id" --jq '.data.updateIssueIssueType.issue.issueType.name' \
  | xargs -I{} echo "  type: {}"
}

apply_pr_label() {
  local number="$1"
  local pr_title
  pr_title=$(gh pr view "$number" --repo "$REPO" --json title --jq .title)
  echo "PR #$number: $pr_title"

  # PR 이 닫는 이슈 번호 추출 — closingIssuesReferences 활용 (gemini 피드백 반영 — PR #239)
  local closing_issue
  closing_issue=$(gh pr view "$number" --repo "$REPO" \
    --json closingIssuesReferences --jq '.closingIssuesReferences[0].number // empty')

  local label
  if [[ -n "$closing_issue" ]]; then
    echo "  closing issue: #$closing_issue"
    local issue_title
    issue_title=$(gh issue view "$closing_issue" --repo "$REPO" --json title --jq .title)
    local meta
    meta=$(resolve_label_and_type "$issue_title")
    label="${meta%% *}"
  else
    # 닫는 이슈가 없으면 PR title prefix 로 직접 추론
    echo "  (no closing issue found, inferring from PR title)"
    local meta
    meta=$(resolve_label_and_type "$pr_title")
    label="${meta%% *}"
  fi

  # label 적용 — +를 ,로 변환해 한 번에 전달
  gh pr edit "$number" --repo "$REPO" --add-label "${label//+/,}"
  echo "  label: ${label//+/,}"
}

[[ $# -lt 2 ]] && usage

SUBCOMMAND="$1"
NUMBER="$2"

case "$SUBCOMMAND" in
  issue) apply_issue_label_and_type "$NUMBER" ;;
  pr)    apply_pr_label "$NUMBER" ;;
  *)     usage ;;
esac
