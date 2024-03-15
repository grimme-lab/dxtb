"""
Math: Einsum
============

This module provides a wrapper for the `einsum` function from `opt_einsum`
package. If `opt_einsum` is not installed, it falls back to the `torch.einsum`.
"""

from __future__ import annotations

import torch
from tad_mctc.typing import Any, Tensor

from .._types import wraps

__all__ = [
    "eigh",
    "qr",
    "einsum",
    "einsum_greedy",
    "einsum_optimal",
]


try:
    from functools import partial

    from opt_einsum import contract

    from dxtb.constants.defaults import EINSUM_OPTIMIZE

    @wraps(contract)
    def _torch_einsum(*args: Any, optimize: Any = EINSUM_OPTIMIZE) -> Tensor:
        f = partial(contract, backend="torch", optimize=optimize)
        return f(*args)  # type: ignore

    @wraps(contract)
    def einsum_greedy(*args: Any) -> Tensor:
        return partial(_torch_einsum, optimize="greedy")(*args)

    @wraps(contract)
    def einsum_optimal(*args: Any) -> Tensor:
        return partial(_torch_einsum, optimize="optimal")(*args)

    @wraps(contract)
    def einsum(*args: Any, optimize: Any = EINSUM_OPTIMIZE) -> Tensor:
        if optimize == "greedy":
            return einsum_greedy(*args)

        if optimize == "optimal":
            return einsum_optimal(*args)

        return _torch_einsum(*args, optimize=optimize)

except ImportError:

    @wraps(torch.einsum)
    def einsum(*args: Any, optimize: Any = None) -> Tensor:
        if optimize is not None:
            from warnings import warn

            warn("Optimization not supported without 'opt_einsum' package.")

        return torch.einsum(*args)

    einsum_optimal = einsum_greedy = einsum


def eigh(matrix: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
    """
    Typed wrapper for PyTorch's `torch.linalg.eigh` function.

    Parameters
    ----------
    matrix : torch.Tensor
        Input matrix

    Returns
    -------
    tuple[Tensor, Tensor]
        Eigenvalues and eigenvectors of the input matrix.
    """
    return torch.linalg.eigh(matrix, *args, **kwargs)


def qr(matrix: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
    """
    Typed wrapper for PyTorch's `torch.linalg.qs` function.

    Parameters
    ----------
    matrix : torch.Tensor
        Input matrix

    Returns
    -------
    tuple[Tensor, Tensor]
        Orthogonal matrix and upper triangular matrix of the input matrix.
    """
    return torch.linalg.qr(matrix, *args, **kwargs)
