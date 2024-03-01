"""
Math: Einsum
============

This module provides a wrapper for the `einsum` function from `opt_einsum`
package. If `opt_einsum` is not installed, it falls back to the `torch.einsum`.
"""

from __future__ import annotations

__all__ = ["einsum"]


try:
    from functools import partial

    from opt_einsum import contract

    from .._types import Any, Tensor, wraps

    @wraps(contract)
    def einsum(*args: Any) -> Tensor:
        return partial(contract, backend="torch")(*args)  # type: ignore

except ImportError:
    import torch

    einsum = torch.einsum
