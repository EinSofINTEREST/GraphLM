"""Growth operators (Phase 1: Net2DeeperNet identity init)."""

from graphlm.growth.function_preservation import assert_function_preserving
from graphlm.growth.net2deeper import add_layer_function_preserving

__all__ = ["add_layer_function_preserving", "assert_function_preserving"]
