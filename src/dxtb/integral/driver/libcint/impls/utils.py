# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from dxtb.typing import Any, Callable, TypeVar

__all__ = ["NDIM", "np2ctypes", "int2ctypes"]

NDIM = 3
"""Number of cartesian dimensions."""

T = TypeVar("T")


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
    return ctypes.c_int(int(a))


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
