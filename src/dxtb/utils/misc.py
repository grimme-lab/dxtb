"""
Utility
=======

Collection of utility functions.
"""
from __future__ import annotations

import torch
import sys
import os

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


class HiddenPrints:
    """Context manager allowing to hide print statements during development."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
