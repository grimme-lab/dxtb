"""
Utility
=======

Collection of utility functions.
"""
from __future__ import annotations

import torch

from .._types import Any, Tensor, TypeGuard
from ..constants import ATOMIC_NUMBER


def is_str_list(x: list[Any]) -> TypeGuard[list[str]]:
    """
    Determines whether all objects in the list are strings.

    Parameters
    ----------
    x : list[Any]
        List to check.

    Returns
    -------
    TypeGuard[list[str]]
        `True` if all objects are strings, `False` otherwise.
    """
    return all(isinstance(i, str) for i in x)


def is_int_list(x: list[Any]) -> TypeGuard[list[int]]:
    """
    Determines whether all objects in the list are integers.

    Parameters
    ----------
    x : list[Any]
        List to check.

    Returns
    -------
    TypeGuard[list[int]]
        `True` if all objects are integers, `False` otherwise.
    """
    return all(isinstance(i, int) for i in x)


def symbol2number(sym_list: list[str]) -> Tensor:
    return torch.flatten(torch.tensor([ATOMIC_NUMBER[s.title()] for s in sym_list]))


def symmetrize(x: Tensor) -> Tensor:
    """
    Symmetrize a tensor after checking if it is symmetric within a threshold.

    Parameters
    ----------
    x : Tensor
        Tensor to check and symmetrize.

    Returns
    -------
    Tensor
        Symmetrized tensor.

    Raises
    ------
    RuntimeError
        If the tensor is not symmetric within the threshold.
    """
    atol = torch.finfo(x.dtype).eps * 10
    if not torch.allclose(x, x.mT, atol=atol):
        raise RuntimeError(
            f"Matrix appears to be not symmetric (atol={atol}, dtype={x.dtype})."
        )

    return (x + x.mT) / 2
