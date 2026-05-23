#!/usr/bin/env bash
# lint.sh — 프로젝트 루트 기준으로 ruff + nbqa 를 실행합니다.
#
# 사용법:
#   scripts/lint.sh           # 전체 lint (src/, tests/, notebooks/)
#   scripts/lint.sh src/      # 명시적 경로 지정
#   scripts/lint.sh --fix     # 자동 수정 가능한 항목 수정
#
# 호출 경로에 관계없이 항상 프로젝트 루트(pyproject.toml 위치)에서 실행됩니다.
# Makefile 의 `lint` 타겟이 이 스크립트를 호출합니다.

set -euo pipefail

# pyproject.toml 이 있는 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v ruff &>/dev/null; then
  echo "ruff not found. Install with:" >&2
  echo "  pip install ruff" >&2
  echo "  # or via project dev deps: pip install -e '.[dev]'" >&2
  exit 1
fi

# --fix 등 인자는 ruff 에 그대로 전달
RUFF_ARGS=("$@")

echo "Running ruff check in $PROJECT_ROOT ..."
ruff check "${RUFF_ARGS[@]:-.}"

echo "Running ruff format --check ..."
# --fix 모드면 format 도 적용, 아니면 check 만
if [[ " ${RUFF_ARGS[*]:-} " == *" --fix "* ]]; then
  ruff format .
else
  ruff format --check .
fi

# 노트북이 존재할 때만 nbqa 적용 (선택적 의존성)
if [[ -d notebooks ]] && command -v nbqa &>/dev/null; then
  echo "Running nbqa ruff notebooks/ ..."
  nbqa ruff notebooks/
fi

# 논문 요약 frontmatter 검증 (docs/papers/ 가 존재할 때)
if [[ -d docs/papers ]]; then
  echo "Running check-papers-frontmatter.py ..."
  python3 scripts/check-papers-frontmatter.py
fi
