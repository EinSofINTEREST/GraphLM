# GraphLM Development Rules

이 디렉토리는 GraphLM(그래프 + 언어 모델 연구·실험) 프로젝트의 개발 규칙입니다.
주 언어는 Python, 실험은 Jupyter Notebook(`.ipynb`)입니다.

## 개요

**GraphLM** 은 다음을 목표로 합니다:

- 그래프 구조와 언어 모델을 결합한 연구·실험 진행
- 재현 가능한 (reproducible) 실험 코드베이스 구축
- 패키지 코드(`src/graphlm/`) ↔ 노트북(`notebooks/`) 분리

## Ruleset 구조

| 파일 | 주제 |
|---|---|
| [01-architecture.md](01-architecture.md) | 시스템 / 디렉토리 구조 / 재현성 원칙 |
| [04-error-handling.md](04-error-handling.md) | 예외 타입 / 로깅 / NaN 감지 / 결정론 |
| [05-testing.md](05-testing.md) | pytest 패턴 / fixture / 마커 / 커버리지 |
| [06-code-style.md](06-code-style.md) | PEP 8 / ruff / 네이밍 / docstring / 커밋 컨벤션 |
| [07-workflow.md](07-workflow.md) | AI 자율 진행 / commit-per-TODO / PR 자동 / Label/Type |

> **참고**: 본 프로젝트는 production 크롤러/파이프라인이 아닌 연구 프로젝트입니다.
> 기존 ruleset 중 도메인 특화 문서(`02-crawler-implementation.md`,
> `03-data-processing.md`)는 제거되었습니다 — 필요 시 git history 참조.

## Quick Start

**새 작업자라면:**

1. 진입점: 루트 [`CLAUDE.md`](../../CLAUDE.md) (목차)
2. 아키텍처 이해: [01-architecture.md](01-architecture.md)
3. 코드 스타일: [06-code-style.md](06-code-style.md)
4. AI 협업 워크플로: [07-workflow.md](07-workflow.md)

**작업 시작 시:**

- 새 기능 → [07-workflow.md](07-workflow.md) 의 issue-first 정책 → [06-code-style.md](06-code-style.md) + [05-testing.md](05-testing.md)
- 버그 수정 → [04-error-handling.md](04-error-handling.md) + 회귀 테스트
- 노트북 실험 → [01-architecture.md](01-architecture.md) 의 코드/노트북 분리 원칙 준수

## 핵심 원칙

1. **재현성** — 시드 고정, 의존성 버전 명시, 결과 저장 표준화
2. **단순성** — 연구 코드는 읽기 쉬워야 한다. 과도한 추상화 금지
3. **분리** — 로직은 `src/graphlm/`, 실험 흐름은 `notebooks/`
4. **테스트** — 패키지 코드는 pytest 로 검증. 노트북은 정적 lint 만
5. **컨벤션** — 커밋·PR·이슈 prefix 3분리 체계 ([07-workflow.md](07-workflow.md) 규약 6)

## CI 게이트

다음은 CI 가 자동으로 강제합니다:

| 게이트 | 도구 | 강제 위치 |
|---|---|---|
| 커밋 메시지 형식 | regex | `.github/workflows/ci-convention.yml` |
| PR 타이틀 형식 | regex | `.github/workflows/ci-convention.yml` |
| 코드 포맷 | `ruff format --check` | `.github/workflows/ci-quality.yml` |
| 정적 분석 | `ruff check` + `nbqa` | `.github/workflows/ci-quality.yml` |
| 테스트 + 커버리지 70% | `pytest --cov` | `.github/workflows/ci-quality.yml` |
| Linked Issue | GraphQL | `.github/workflows/ci-convention.yml` |

상세는 [`docs/ci/conventions.md`](../../docs/ci/conventions.md) 와
[`docs/ci/status-checks.md`](../../docs/ci/status-checks.md) 참조.
