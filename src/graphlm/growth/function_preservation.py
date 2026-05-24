"""Function preservation verifier — Net2Net 의 핵심 invariant.

학습 중 growth operator 가 \"확장 직후 forward output 동일\" 을 위반하면 학습 spike 발생.
본 helper 는 unit test + (optional) 학습 runtime 의 safety check 로 사용.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

# Re-export from utils.exceptions so existing imports keep working while
# inheritance is unified under GraphLMError (per .claude/rules/04-error-handling.md).
from graphlm.utils.exceptions import FunctionPreservationError

__all__ = ["FunctionPreservationError", "assert_function_preserving"]


def assert_function_preserving(
    grow_op: Callable[[], None],
    forward_fn: Callable[[Tensor], Tensor],
    sample_input: Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Run `grow_op` and assert that `forward_fn(sample_input)` is unchanged.

    Args:
        grow_op: Callable that performs the growth (e.g. ``lambda: add_layer(model)``).
            Called once between the two forward passes.
        forward_fn: Function that takes the sample input and returns the model output.
            Typically ``model.forward`` or a partial of it.
        sample_input: Tensor input shared by both forward passes.
        rtol / atol: tolerance for ``torch.allclose``.

    Raises:
        FunctionPreservationError: if outputs differ beyond the tolerance.

    Note:
        Caller should set the model to eval mode (no dropout / batch norm running stats)
        before invoking — otherwise stochastic ops can give false positives.
    """
    with torch.no_grad():
        out_before = forward_fn(sample_input).clone()
    grow_op()
    with torch.no_grad():
        out_after = forward_fn(sample_input)
    if not torch.allclose(out_before, out_after, rtol=rtol, atol=atol):
        max_diff = (out_before - out_after).abs().max().item()
        raise FunctionPreservationError(
            f"Growth op broke function preservation: max abs diff = {max_diff:.3e} "
            f"(rtol={rtol}, atol={atol})"
        )
