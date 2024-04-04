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
# pylint: disable=protected-access
"""
Utility: Miscellaneous
======================

Collection of miscellaneous utility functions containing:
- Type guards
- JIT enabler
- Memoization decorators
"""
from __future__ import annotations

from functools import wraps

import torch

from dxtb.typing import Any, Callable, Tensor, TypeGuard, TypeVar

from ..basis.types import AtomCGTOBasis

T = TypeVar("T")


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
    if not isinstance(x, list):
        return False
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
    if not isinstance(x, list):
        return False
    return all(isinstance(i, int) for i in x)


def is_basis_list(x: Any) -> TypeGuard[list[AtomCGTOBasis]]:
    """
    Determines whether all objects in the list are `AtomCGTOBasis`.

    Parameters
    ----------
    x : list[Any]
        List to check.

    Returns
    -------
    TypeGuard[list[AtomCGTOBasis]]
        `True` if all objects are `AtomCGTOBasis`, `False` otherwise.
    """
    if not isinstance(x, list):
        return False
    return all(isinstance(i, AtomCGTOBasis) for i in x)


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
    Memoization with shared cache among all instances of the decorated function.
    This decorator allows specification of `__slots__`. It works with and
    without function arguments.

    Note that `lru_cache` can produce memory leaks for a method.

    Parameters
    ----------
    fcn : Callable[[Any], T]
        Function to memoize

    Returns
    -------
    Callable[[Any], T]
        Memoized function.
    """

    # creating the cache outside the wrapper shares it across instances
    cache = {}

    @wraps(fcn)
    def wrapper(self, *args, **kwargs):
        # create unique key for all instances in cache dictionary
        key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

        # if the result is already in the cache, return it
        if key in cache:
            return cache[key]

        # if key is not found, compute the result
        result = fcn(self, *args, **kwargs)
        cache[key] = result
        return result

    def clear():
        cache.clear()

    def get():
        return cache

    setattr(wrapper, "clear", clear)
    setattr(wrapper, "clear_cache", clear)
    setattr(wrapper, "get_cache", get)

    return wrapper


def memoizer(fcn: Callable[..., T]) -> Callable[..., T]:  # pragma: no cover
    """
    Memoization decorator that writes the cache to the object itself, hence not
    allowing the specification of `__slots__`. It works with and without
    function arguments.

    Note that `lru_cache` can produce memory leaks for a method.

    Parameters
    ----------
    fcn : Callable[[Any], T]
        Function to memoize

    Returns
    -------
    Callable[[Any], T]
        Memoized function.
    """

    @wraps(fcn)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_cache"):
            # Create a cache dictionary as an instance attribute
            self._cache = {}

        # Create a unique key for the cache dictionary
        key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

        # If the result is already in the cache, return it
        if key in self._cache:
            return self._cache[key]

        # If key is not found, compute the result
        result = fcn(self, *args, **kwargs)
        self._cache[key] = result
        return result

    return wrapper


def dependent_memoize(*dependency_getters: Callable[..., Any]):
    """
    Memoization with multiple dependency-based cache invalidation. This
    decorator allows specification of `__slots__`. It works with and without
    function arguments.

    Warning
    -------
    This is an experimental feature, which can cause memory leaks!
    """

    def decorator(fcn: Callable[..., T]) -> Callable[..., T]:
        # creating the cache outside the wrapper shares it across instances
        cache = {}
        dependency_cache = {}

        @wraps(fcn)
        def wrapper(self, *args, **kwargs):
            # create unique key for all instances in cache dictionary
            key = (id(self), fcn.__name__, args, frozenset(kwargs.items()))

            # get current deps
            current_deps = tuple(getter(self) for getter in dependency_getters)
            cached_deps = dependency_cache.get(key)

            # Check if the cache has been invalidated
            cache_invalidated = False
            if cached_deps is None or len(cached_deps) != len(current_deps):
                cache_invalidated = True
            else:
                for curr, cached in zip(current_deps, cached_deps):
                    if not torch.equal(curr, cached):
                        cache_invalidated = True
                        break

            if not cache_invalidated and key in cache:
                return cache[key]

            # if the result is not in cache or deps have changed, compute the result
            result = fcn(self, *args, **kwargs)
            cache[key] = result
            dependency_cache[key] = current_deps
            return result

        def clear():
            cache.clear()
            dependency_cache.clear()

        def get():
            return cache

        def get_dep():
            return dependency_cache

        setattr(wrapper, "clear", clear)
        setattr(wrapper, "clear_cache", clear)
        setattr(wrapper, "get_cache", get)
        setattr(wrapper, "get_dep_cache", get_dep)

        return wrapper

    return decorator
