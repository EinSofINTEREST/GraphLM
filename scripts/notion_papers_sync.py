#!/usr/bin/env python3
"""Sync paper md summaries under docs/papers/ to the Notion 'Papers' data source.

Modes:
  - file     : sync a single paper file (--path docs/papers/.../xxx.md)
  - backfill : delete all rows then re-create from every paper md under docs/papers/

Required env:
  NOTION_API_TOKEN  Notion integration token (workspace bot)
  PAPERS_DS_ID      Notion data source id (defaults to constant below)
  GITHUB_BLOB_BASE  Optional GitHub blob URL base for "전체 본문 보기" link

Notion 'Papers' data source schema (a71d6289-d13a-4d4a-a66d-ff52df038939):
  제목 (title) · 별칭 (rich_text) · 연도 (number) · venue (select)
  카테고리 (select: graph / hybrid / computation-graph)
  갈래 (multi_select) · paradigm 정렬 (select) · status (select)
  URL (url) · 코드 (url) · Local 경로 (rich_text)

This script reuses the markdown→Notion-blocks conversion logic from
the IssueTracker `notion_pr_sync.py` (issue #21).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"
DEFAULT_DS_ID = "a71d6289-d13a-4d4a-a66d-ff52df038939"
DEFAULT_GITHUB_BLOB_BASE = "https://github.com/EinSofINTEREST/GraphLM/blob/main"

PAPERS_DIR = Path("docs/papers")
EXCLUDED_FILES: frozenset[str] = frozenset({"README.md", "_template.md"})

# Mapping: paper venue string in frontmatter → Notion select option name
VENUE_MAP: dict[str, str] = {
    "ICLR 2016": "ICLR",
    "ICLR 2017": "ICLR",
    "ICLR 2018": "ICLR",
    "ICLR 2019": "ICLR",
    "ICLR 2020": "ICLR",
    "ICLR 2021": "ICLR",
    "ICLR 2022": "ICLR",
    "ICLR 2023": "ICLR",
    "ICLR 2024": "ICLR",
    "NeurIPS 2017": "NeurIPS",
    "NeurIPS 2020": "NeurIPS",
    "NeurIPS 2021": "NeurIPS",
    "NeurIPS 2023": "NeurIPS",
    "ICML 2019": "ICML",
    "ICML 2020": "ICML",
    "ICML 2023": "ICML",
    "ICML 2024": "ICML",
    "JMLR 2022": "JMLR",
    "AAAI 2020": "AAAI",
    "ACL 2022": "ACL",
    "ACL 2024": "ACL",
    "ICCV 2021": "ICCV",
    "CVPR 2018": "CVPR",
    "Nature Communications 2018": "Nature Comm",
}

# Default to "arXiv preprint" if not in map
DEFAULT_VENUE = "arXiv preprint"

# Mapping: directory under docs/papers/ → category select option
CATEGORY_MAP: dict[str, str] = {
    "graph": "graph",
    "hybrid": "hybrid",
    "computation-graph": "computation-graph",
}

# Mapping: paper tag (frontmatter) → 갈래 multi_select option
# Only tags that map to known options are kept; others are dropped.
TAG_TO_BRANCH: dict[str, str] = {
    # Growing
    "growing": "Growing",
    "net2net": "Growing",
    "bert2bert": "Growing",
    "ligo": "Growing",
    "msg": "Growing",
    "stacking": "Growing",
    # DST
    "dst": "DST",
    "sparse": "DST",
    "set": "DST",
    "rigl": "DST",
    "top-kast": "DST",
    "sparsegpt": "DST",
    "wanda": "DST",
    # DARTS
    "darts": "DARTS",
    # Adaptive Trigger
    "adaptive-trigger": "Adaptive Trigger",
    "den": "Adaptive Trigger",
    "autogrow": "Adaptive Trigger",
    "firefly": "Adaptive Trigger",
    "self-expanding": "Adaptive Trigger",
    "gradmax": "Adaptive Trigger",
    # Resource·Deployment
    "resource-aware": "Resource Deployment",
    "morphnet": "Resource Deployment",
    "deployment": "Resource Deployment",
    "ofa": "Resource Deployment",
    "matformer": "Resource Deployment",
    "layerskip": "Resource Deployment",
    "mobilellm": "Resource Deployment",
    "sheared-llama": "Resource Deployment",
    # Sparse Activation (MoE 류)
    "moe": "Sparse Activation",
    "gshard": "Sparse Activation",
    "switch-transformer": "Sparse Activation",
    "mixtral": "Sparse Activation",
    "mod": "Sparse Activation",
    "universal-transformer": "Sparse Activation",
    # Architecture-as-graph
    "autoformer": "Architecture-as-graph",
    "ghn": "Architecture-as-graph",
    "ghn3": "Architecture-as-graph",
    "nas": "Architecture-as-graph",
    # GNN
    "gnn": "GNN",
    "gcn": "GNN",
    "gat": "GNN",
    "graphsage": "GNN",
    "graphormer": "GNN",
    # data-as-graph
    "graph": "data-as-graph",
    # hybrid (Graph + LM)
    "hybrid": "hybrid",
    "graph-bert": "hybrid",
    "graphformers": "hybrid",
    "glem": "hybrid",
    "graphgpt": "hybrid",
}

# Paradigm 정렬: directory + tags 로 결정
PARADIGM_BY_TAG: dict[str, str] = {
    "dynamic-param": "dynamic param count",
    "sparse-activation": "sparse activation",
    "architecture-as-graph": "architecture-as-graph",
    "data-as-graph": "data-as-graph",
}


# ---------- helpers ----------


def log(msg: str) -> None:
    print(msg, flush=True)


def notion_request(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    token: str | None = None,
) -> dict:
    token = token or os.environ["NOTION_API_TOKEN"]
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{NOTION_API}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        method=method,
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(2**attempt)
                continue
            body_text = e.read().decode(errors="replace")[:800]
            raise RuntimeError(f"Notion {method} {path} -> {e.code}: {body_text}") from e
        except urllib.error.URLError:
            if attempt < 3:
                time.sleep(2**attempt)
                continue
            raise
    return None


def rt(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    code: bool = False,
    link: str | None = None,
) -> dict:
    return {
        "type": "text",
        "text": {
            "content": text,
            "link": ({"url": link} if link else None),
        },
        "annotations": {
            "bold": bold,
            "italic": italic,
            "strikethrough": False,
            "underline": False,
            "code": code,
            "color": "default",
        },
    }


# ---------- markdown → Notion blocks (subset, adapted from IssueTracker) ----------

_INLINE_RE = re.compile(
    r"\*\*(.+?)\*\*"
    r"|`([^`\n]+?)`"
    r"|\[([^\]\n]+?)\]\(([^)\s]+?)\)"
    r"|\*([^*\n]+?)\*"
)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_LIST_RE = re.compile(r"^[-*+]\s+(.+)$")
_NUMLIST_RE = re.compile(r"^\d+\.\s+(.+)$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _truncate(s: str, n: int = 1900) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def md_inline(text: str) -> list[dict]:
    spans: list[dict] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            spans.append(rt(_truncate(text[pos : m.start()])))
        if m.group(1) is not None:
            spans.append(rt(_truncate(m.group(1)), bold=True))
        elif m.group(2) is not None:
            spans.append(rt(_truncate(m.group(2)), code=True))
        elif m.group(3) is not None and m.group(4) is not None:
            url = m.group(4)
            if url.startswith(("http://", "https://", "mailto:")):
                spans.append(rt(_truncate(m.group(3)), link=url))
            else:
                spans.append(rt(_truncate(m.group(3))))
        elif m.group(5) is not None:
            spans.append(rt(_truncate(m.group(5)), italic=True))
        pos = m.end()
    if pos < len(text):
        spans.append(rt(_truncate(text[pos:])))
    return spans or [rt("")]


def _block(t: str, **payload) -> dict:
    return {"object": "block", "type": t, t: payload}


def md_to_blocks(md: str) -> list[dict]:
    """Convert markdown body to Notion block list."""
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].rstrip()
        ls = s.lstrip()

        # code fence
        if ls.startswith("```"):
            lang = ls[3:].strip().lower() or "plain text"
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].lstrip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(
                _block(
                    "code",
                    rich_text=[rt(_truncate("\n".join(code_lines)))],
                    language=lang
                    if lang in {"python", "shell", "json", "yaml", "markdown"}
                    else "plain text",
                )
            )
            continue

        if not s.strip():
            i += 1
            continue

        m = _HEADING_RE.match(s)
        if m:
            level = len(m.group(1))
            blocks.append(_block(f"heading_{level}", rich_text=md_inline(m.group(2))))
            i += 1
            continue

        if _HR_RE.match(s):
            blocks.append(_block("divider"))
            i += 1
            continue

        if s.startswith(">"):
            q_lines: list[str] = []
            while i < n and lines[i].rstrip().startswith(">"):
                q_lines.append(re.sub(r"^>\s?", "", lines[i].rstrip()))
                i += 1
            blocks.append(_block("quote", rich_text=md_inline("\n".join(q_lines))))
            continue

        m = _LIST_RE.match(s)
        if m:
            blocks.append(_block("bulleted_list_item", rich_text=md_inline(m.group(1))))
            i += 1
            continue
        m = _NUMLIST_RE.match(s)
        if m:
            blocks.append(_block("numbered_list_item", rich_text=md_inline(m.group(1))))
            i += 1
            continue

        # paragraph
        para = [lines[i]]
        i += 1
        while i < n:
            nxt = lines[i].rstrip()
            if not nxt.strip():
                break
            if nxt.lstrip().startswith(("```", ">")):
                break
            if _HEADING_RE.match(nxt) or _HR_RE.match(nxt):
                break
            if _LIST_RE.match(nxt) or _NUMLIST_RE.match(nxt):
                break
            para.append(lines[i])
            i += 1
        text = " ".join(line.strip() for line in para if line.strip())
        if text:
            blocks.append(_block("paragraph", rich_text=md_inline(text)))
    return blocks


# ---------- paper md parsing ----------


def parse_paper(path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_md)."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text, re.DOTALL)
    if not m:
        raise ValueError(f"{path}: frontmatter not found")

    fm_block, body = m.group(1), m.group(2)
    fm: dict[str, str | list[str]] = {}

    for line in fm_block.splitlines():
        if line.startswith(("  ", "\t")) or not line.strip():
            continue
        km = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if not km:
            continue
        key = km.group(1)
        value = re.split(r"\s+#", km.group(2), maxsplit=1)[0].strip()
        # strip surrounding quotes
        value_stripped = value.strip('"').strip("'")
        # parse list literal: ["a", "b", "c"]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                fm[key] = []
            else:
                items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
                fm[key] = [s for s in items if s]
        else:
            fm[key] = value_stripped
    return fm, body


def determine_paradigm(category: str, tags: list[str]) -> str:
    """Decide '패러다임 정렬' from category + tags."""
    if "dynamic-param" in tags:
        return "dynamic param count"
    if category == "computation-graph":
        # sparse activation / architecture-as-graph 만 dynamic-param 외
        if any(
            t in tags
            for t in (
                "moe",
                "mod",
                "universal-transformer",
                "switch-transformer",
                "gshard",
                "mixtral",
            )
        ):
            return "sparse activation"
        if any(t in tags for t in ("autoformer", "ghn", "ghn3", "nas")):
            return "architecture-as-graph"
        # default for computation-graph 미해당 = dynamic
        return "dynamic param count"
    if category in {"graph", "hybrid"}:
        return "data-as-graph"
    return "data-as-graph"


def determine_branches(tags: list[str]) -> list[str]:
    branches: set[str] = set()
    for t in tags:
        if t in TAG_TO_BRANCH:
            branches.add(TAG_TO_BRANCH[t])
    return sorted(branches)


def determine_short_id(path: Path) -> str:
    """Extract short id from filename: 2017-gcn-kipf.md → GCN (or first hyphen-separated middle tokens uppercased)."""
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        return stem.upper()
    # year-<short-id-tokens...>-author
    mid_tokens = parts[1:-1]
    return "-".join(t.upper() for t in mid_tokens)


def paper_to_props(path: Path, fm: dict, github_blob_base: str) -> dict:
    title = fm.get("title", path.stem)
    if isinstance(title, list):
        title = " ".join(title)
    title = (title or path.stem).strip('"').strip("'")

    short_id = determine_short_id(path)
    year_raw = fm.get("year", "")
    try:
        year = int(str(year_raw))
    except (TypeError, ValueError):
        year = None

    venue_raw = fm.get("venue", "")
    if isinstance(venue_raw, list):
        venue_raw = " ".join(venue_raw)
    venue_raw = str(venue_raw).strip('"').strip("'")
    venue = VENUE_MAP.get(venue_raw, DEFAULT_VENUE)

    # category from path: docs/papers/<category>/file.md
    rel = path.relative_to(PAPERS_DIR)
    category = rel.parts[0] if rel.parts else "computation-graph"
    category = CATEGORY_MAP.get(category, category)

    tags = fm.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    tags = [t.lower().strip() for t in tags if t]

    branches = determine_branches(tags)
    paradigm = determine_paradigm(category, tags)

    status_raw = fm.get("status", "draft")
    if isinstance(status_raw, list):
        status_raw = status_raw[0] if status_raw else "draft"
    status = str(status_raw).strip('"').strip("'") or "draft"

    url = fm.get("url", "")
    if isinstance(url, list):
        url = url[0] if url else ""
    url = str(url).strip('"').strip("'")

    code_url = fm.get("code_url", "")
    if isinstance(code_url, list):
        code_url = code_url[0] if code_url else ""
    code_url = str(code_url).strip('"').strip("'")

    local_path = str(rel.as_posix())
    local_path_full = f"docs/papers/{local_path}"

    props: dict[str, dict] = {
        "제목": {"title": [rt(title)]},
        "별칭": {"rich_text": [rt(short_id)]},
        "카테고리": {"select": {"name": category}},
        "venue": {"select": {"name": venue}},
        "갈래": {"multi_select": [{"name": b} for b in branches]},
        "paradigm 정렬": {"select": {"name": paradigm}},
        "status": {"select": {"name": status}},
        "Local 경로": {"rich_text": [rt(local_path_full)]},
    }
    if year is not None:
        props["연도"] = {"number": year}
    if url:
        props["URL"] = {"url": url}
    if code_url:
        props["코드"] = {"url": code_url}
    return props


def paper_to_body_blocks(path: Path, fm: dict, body: str, github_blob_base: str) -> list[dict]:
    """Build Notion blocks: header callout + body markdown + footer link."""
    blocks: list[dict] = []
    rel = path.relative_to(PAPERS_DIR)
    local_path_full = f"docs/papers/{rel.as_posix()}"
    github_url = f"{github_blob_base}/{local_path_full}"

    # Header callout — title + meta
    title = fm.get("title", "")
    if isinstance(title, list):
        title = " ".join(title)
    title = str(title).strip('"').strip("'")

    authors = fm.get("authors", "")
    if isinstance(authors, list):
        authors = ", ".join(authors)
    authors = str(authors).strip('"').strip("'")

    venue = str(fm.get("venue", "")).strip('"').strip("'")
    str(fm.get("year", "")).strip('"').strip("'")

    blocks.append(
        _block(
            "callout",
            rich_text=[
                rt(f"{title}\n", bold=True),
                rt(f"{authors} · {venue}" if venue else f"{authors}"),
            ],
            icon={"type": "emoji", "emoji": "📄"},
            color="gray_background",
        )
    )

    # Body markdown
    blocks.extend(md_to_blocks(body))

    # Footer
    blocks.append(_block("divider"))
    blocks.append(
        _block(
            "paragraph",
            rich_text=[
                rt("원본 md: "),
                rt(local_path_full, code=True, link=github_url),
            ],
        )
    )
    return blocks


# ---------- Notion ops ----------


def iter_all_rows(ds_id: str):
    cursor = None
    while True:
        payload = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        res = notion_request(f"/data_sources/{ds_id}/query", "POST", payload)
        for r in res.get("results", []):
            yield r["id"]
        if not res.get("has_more"):
            return
        cursor = res.get("next_cursor")


def archive_page(page_id: str) -> None:
    notion_request(f"/pages/{page_id}", "PATCH", {"in_trash": True})


def replace_page_body(page_id: str, new_blocks: list[dict]) -> None:
    """Delete existing children then append new blocks in chunks of 100."""
    cursor: str | None = None
    existing_ids: list[str] = []
    while True:
        path = f"/blocks/{page_id}/children?page_size=100"
        if cursor:
            path += f"&start_cursor={cursor}"
        res = notion_request(path, "GET")
        for b in res.get("results", []):
            existing_ids.append(b["id"])
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")

    for bid in existing_ids:
        notion_request(f"/blocks/{bid}", "DELETE")
        time.sleep(0.15)

    for start in range(0, len(new_blocks), 100):
        chunk = new_blocks[start : start + 100]
        notion_request(f"/blocks/{page_id}/children", "PATCH", {"children": chunk})
        time.sleep(0.2)


def create_paper_row(ds_id: str, path: Path, github_blob_base: str) -> str:
    fm, body = parse_paper(path)
    props = paper_to_props(path, fm, github_blob_base)
    res = notion_request(
        "/pages",
        "POST",
        {"parent": {"data_source_id": ds_id}, "properties": props},
    )
    page_id = res["id"]
    replace_page_body(page_id, paper_to_body_blocks(path, fm, body, github_blob_base))
    return page_id


# ---------- modes ----------


def mode_file(ds_id: str, path: Path, github_blob_base: str) -> int:
    if not path.exists():
        log(f"error: {path} not found")
        return 2
    page_id = create_paper_row(ds_id, path, github_blob_base)
    log(f"created row {page_id} for {path}")
    return 0


def _existing_short_ids(ds_id: str) -> set[str]:
    """Return set of '별칭' (short id) values for existing rows."""
    existing: set[str] = set()
    cursor: str | None = None
    while True:
        payload = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        res = notion_request(f"/data_sources/{ds_id}/query", "POST", payload)
        for r in res.get("results", []):
            props = r.get("properties", {})
            alias = props.get("별칭", {}).get("rich_text", [])
            if alias and alias[0].get("plain_text"):
                existing.add(alias[0]["plain_text"])
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return existing


def mode_backfill(ds_id: str, github_blob_base: str, skip_existing: bool = False) -> int:
    paper_files = sorted(p for p in PAPERS_DIR.rglob("*.md") if p.name not in EXCLUDED_FILES)
    if not paper_files:
        log("no paper md files found; aborting")
        return 2

    if skip_existing:
        existing = _existing_short_ids(ds_id)
        log(f"skip-existing mode: {len(existing)} rows already in DB")
        paper_files = [p for p in paper_files if determine_short_id(p) not in existing]
        log(f"will create {len(paper_files)} missing rows")
    else:
        log("== archiving existing rows ==")
        archived = 0
        for row_id in iter_all_rows(ds_id):
            archive_page(row_id)
            archived += 1
            time.sleep(0.2)
        log(f"archived {archived} rows")
        log(f"== creating {len(paper_files)} rows ==")

    errors: list[tuple[Path, str]] = []
    for i, path in enumerate(paper_files, 1):
        try:
            page_id = create_paper_row(ds_id, path, github_blob_base)
            if i % 5 == 0 or i == len(paper_files):
                log(f"  {i}/{len(paper_files)}: {path.name} ({page_id})")
        except Exception as exc:  # noqa: BLE001 - 1편 실패가 전체 중단 막기
            log(f"  ERROR {path.name}: {exc}")
            errors.append((path, str(exc)))
        time.sleep(0.3)

    if errors:
        log(f"\n{len(errors)} papers failed:")
        for path, msg in errors:
            log(f"  - {path}: {msg[:120]}")
        return 1
    log("backfill complete")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["file", "backfill"], required=True)
    ap.add_argument("--ds-id", default=os.environ.get("PAPERS_DS_ID", DEFAULT_DS_ID))
    ap.add_argument(
        "--github-blob-base",
        default=os.environ.get("GITHUB_BLOB_BASE", DEFAULT_GITHUB_BLOB_BASE),
    )
    ap.add_argument("--path", help="paper md path (file mode)")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="backfill mode: skip rows already in DB (idempotent resume)",
    )
    args = ap.parse_args(argv)

    if "NOTION_API_TOKEN" not in os.environ:
        log("error: NOTION_API_TOKEN not set")
        return 2

    if args.mode == "file":
        if not args.path:
            log("error: --path required for file mode")
            return 2
        return mode_file(args.ds_id, Path(args.path), args.github_blob_base)
    return mode_backfill(args.ds_id, args.github_blob_base, args.skip_existing)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
