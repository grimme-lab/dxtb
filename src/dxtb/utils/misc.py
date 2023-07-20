# pylint: disable=protected-access
"""
Utility
=======

Collection of utility functions.
"""
from __future__ import annotations

from functools import wraps

import torch

from .._types import Any, Callable, T, Tensor, TypeGuard
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


def convert_float_tensor(
    d: dict[str, float | Tensor],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = torch.tensor(value, device=device, dtype=dtype)

    return d  # type: ignore


def set_jit_enabled(enabled: bool) -> None:
    """
    Enables/disables JIT.

    Parameters
    ----------
    enabled : bool
        State to set JIT to.
    """
    if enabled:
        torch.jit._state.enable()  # type: ignore
    else:
        torch.jit._state.disable()  # type: ignore


def memoize(fcn: Callable[..., T]) -> Callable[..., T]:
    """
    Memoization decorator that allows specification of `__slots__`. It works
    with and without function arguments.

    Note that `lru_cache` can produce memory leak for a method.

    Parameters
    ----------
    fcn : Callable[[Any], T]
        Function to memoize

    Returns
    -------
    Callable[[Any], T]
        Memoized function.
    """
    cache = {}

    @wraps(fcn)
    def wrapper(self, *args, **kwargs):
        # create unique key for cache dictionary
        key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

        # if the result is already in the cache, return it
        if key in cache:
            return cache[key]

        # if key is not found, compute the result
        result = fcn(self, *args, **kwargs)
        cache[key] = result
        return result

    return wrapper
