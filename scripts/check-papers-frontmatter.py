#!/usr/bin/env python3
"""docs/papers/**/*.md 의 frontmatter 필수 키 누락을 검증한다.

`_template.md` 와 인덱스 `README.md` 는 검증에서 제외한다.
PyYAML 의존을 피하기 위해 frontmatter 는 간이 파서로 키 존재 여부만 확인한다.

Usage:
    python scripts/check-papers-frontmatter.py
    python scripts/check-papers-frontmatter.py docs/papers/graph/2017-gcn-kipf.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_KEYS: tuple[str, ...] = (
    "title",
    "authors",
    "year",
    "venue",
    "url",
    "tags",
    "status",
)
ALLOWED_STATUS: tuple[str, ...] = ("draft", "reviewed", "archived")

# 검증 제외 파일명 (basename 기준)
EXCLUDED_FILES: frozenset[str] = frozenset({"README.md", "_template.md"})

PAPERS_DIR = Path("docs/papers")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
KEY_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:")


def parse_frontmatter_keys(text: str) -> dict[str, str]:
    """Frontmatter 블록에서 top-level 키-값을 추출한다.

    값은 빈 문자열이거나 placeholder (예: ``"<...>"``) 인지 후속 판별을 위해 raw 로 보관.
    nested 값은 무시 — 본 lint 는 키 존재 + 비어있지 않음만 검증.
    """
    match = FRONTMATTER_RE.search(text)
    if not match:
        return {}
    block = match.group(1)
    keys: dict[str, str] = {}
    for line in block.splitlines():
        # 들여쓰기된 줄은 리스트/중첩값 — top-level 만 본다
        if line.startswith((" ", "\t")) or not line.strip():
            continue
        m = KEY_LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        # 인라인 주석 (` # ...`) 제거. 따옴표 내부 # 은 보존되지 않지만 frontmatter 값에 거의 없음
        value = re.split(r"\s+#", line[m.end() :], maxsplit=1)[0].strip()
        keys[key] = value
    return keys


def is_placeholder(value: str) -> bool:
    """frontmatter 값이 미작성 placeholder 인지 판정.

    빈 값 외에 `<...>` 형태의 마커가 값 어디든 (리스트 내부 포함) 있으면 placeholder 로 본다.
    예: ``<YYYY>``, ``["<카테고리 키워드들>"]``.
    """
    if not value:
        return True
    stripped = value.strip().strip('"').strip("'")
    if not stripped:
        return True
    return bool(re.search(r"<[^>]+>", value))


def check_file(path: Path) -> list[str]:
    """단일 md 파일 검증. 위반 메시지 리스트 반환 (빈 리스트면 통과)."""
    text = path.read_text(encoding="utf-8")
    keys = parse_frontmatter_keys(text)
    if not keys:
        return [f"{path}: frontmatter 블록(---...---)이 없음"]

    errors: list[str] = []
    for required in REQUIRED_KEYS:
        if required not in keys:
            errors.append(f"{path}: 필수 키 누락 — `{required}`")
        elif is_placeholder(keys[required]):
            errors.append(f"{path}: 키 `{required}` 가 비어있거나 placeholder ({keys[required]!r})")

    status_val = keys.get("status", "").strip().strip('"').strip("'")
    if status_val and status_val not in ALLOWED_STATUS:
        errors.append(f"{path}: status 값 {status_val!r} 허용 외 — {ALLOWED_STATUS} 중 택1")

    return errors


def collect_files(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    if not PAPERS_DIR.exists():
        return []
    return [p for p in sorted(PAPERS_DIR.rglob("*.md")) if p.name not in EXCLUDED_FILES]


def main(argv: list[str]) -> int:
    files = collect_files(argv[1:])
    if not files:
        print("검증할 paper md 파일 없음 (template/README 만 존재). skip.")
        return 0

    all_errors: list[str] = []
    for f in files:
        all_errors.extend(check_file(f))

    if all_errors:
        print("docs/papers frontmatter 검증 실패:\n", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            f"\n총 {len(all_errors)}건 위반. "
            "docs/papers/_template.md 의 frontmatter 키를 참고하세요.",
            file=sys.stderr,
        )
        return 1

    print(f"docs/papers frontmatter 검증 통과 ({len(files)}개 파일).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
