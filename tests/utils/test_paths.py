"""Tests for graphlm.utils.paths."""

from __future__ import annotations

import pytest

from graphlm.utils.paths import repo_root


def test_repo_root_from_cwd_finds_pyproject():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()


def test_repo_root_from_subdirectory():
    root = repo_root()
    sub = root / "src" / "graphlm"
    assert sub.exists(), "test premise: src/graphlm exists"
    assert repo_root(sub) == root


def test_repo_root_accepts_str():
    root = repo_root()
    assert repo_root(str(root)) == root


def test_repo_root_raises_when_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        repo_root(tmp_path)
