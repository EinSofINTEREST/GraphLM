"""패키지 import smoke test."""

from __future__ import annotations


def test_package_imports():
    import graphlm

    assert hasattr(graphlm, "__version__")


def test_version_is_string():
    import graphlm

    assert isinstance(graphlm.__version__, str)
    assert len(graphlm.__version__) > 0
