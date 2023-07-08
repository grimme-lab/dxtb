"""
Utility functions
=================

This module contains helpers required for calculating integrals with libcint.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools

import numpy as np

from ..._types import Any, Callable, TypeVar

T = TypeVar("T")

__all__ = ["NDIM", "np2ctypes", "int2ctypes"]

NDIM = 3
"""Number of cartesian dimensions."""


def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    """
    Convert a numpy array to the corresponding ctypes.

    Parameters
    ----------
    a : np.ndarray
        Array to convert.

    Returns
    -------
    ctypes.c_void_p
        Converted array.
    """
    return a.ctypes.data_as(ctypes.c_void_p)


def int2ctypes(a: int) -> ctypes.c_int:
    """
    Convert a Python integer to a ctypes' integer.

    Parameters
    ----------
    a : int
        Integer to convert.

    Returns
    -------
    ctypes.c_int
        Ctypes' integer.
    """
    return ctypes.c_int(a)


def memoize_method(fcn: Callable[[Any], T]) -> Callable[[Any], T]:
    # alternative for lru_cache for memoizing a method without any arguments
    # lru_cache can produce memory leak for a method
    # this can be known by running test_ks_mem.py individually

    cachename = "__cch_" + fcn.__name__

    @functools.wraps(fcn)
    def new_fcn(self) -> T:
        if cachename in self.__dict__:
            return self.__dict__[cachename]

        res = fcn(self)
        self.__dict__[cachename] = res
        return res

    return new_fcn
