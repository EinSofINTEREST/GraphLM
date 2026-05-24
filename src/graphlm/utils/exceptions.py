"""Base exception classes for GraphLM (per `.claude/rules/04-error-handling.md`).

All custom exceptions in graphlm should inherit from ``GraphLMError`` so that
callers can catch package-specific errors with a single ``except`` clause.
"""

from __future__ import annotations


class GraphLMError(Exception):
    """Base for all GraphLM custom exceptions."""


class FunctionPreservationError(GraphLMError):
    """Raised when a growth operation violates the function-preservation invariant.

    See ``graphlm.growth.function_preservation`` for the assertion helper.
    """
