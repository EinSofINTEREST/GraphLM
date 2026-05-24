"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | str | None = None) -> Path:
    """Return the repository root (directory containing ``pyproject.toml``).

    Walks upward from ``start`` (or ``Path.cwd()``) until a directory
    containing ``pyproject.toml`` is found.

    Args:
        start: Directory to start the search from. Default ``Path.cwd()``.

    Returns:
        Absolute path of the repository root.

    Raises:
        FileNotFoundError: ``pyproject.toml`` not found in any ancestor.
    """
    cur = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for p in (cur, *cur.parents):
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(f"repository root (pyproject.toml) not found from {cur}")
