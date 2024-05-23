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
Calculator: Decorators
======================

Decorators for the Calculator class. These decorators can mark:

- functions that require ``requires_grad=True`` for certain tensors
- functions that require specific interactions to be present
- functions that are computed numerically
- results for caching
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, cast

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions import efield as efield
from dxtb._src.components.interactions.field import efieldgrad as efieldgrad
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Callable, Tensor, TypeVar
from dxtb._src.utils.tensors import tensor_id

if TYPE_CHECKING:
    from ..base import Calculator
del TYPE_CHECKING

__all__ = [
    "requires_positions_grad",
    "requires_efield",
    "requires_efield_grad",
    "requires_efg",
    "requires_efg_grad",
    "numerical",
    "cache",
]

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


def requires_positions_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    @wraps(func)
    def wrapper(
        self: Calculator,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if not positions.requires_grad:
            raise RuntimeError(
                f"Position tensor needs ``requires_grad=True`` in '{func.__name__}'."
            )

        return func(self, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efield(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    @wraps(func)
    def wrapper(
        self: Calculator,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                f"{func.__name__} requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )
        return func(self, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efield_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    @wraps(func)
    def wrapper(
        self: Calculator,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        if not ef.field.requires_grad:
            raise RuntimeError(
                f"Field tensor needs ``requires_grad=True`` in '{func.__name__}'."
            )
        return func(self, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efg(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    @wraps(func)
    def wrapper(
        self: Calculator,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if efieldgrad.LABEL_EFIELD_GRAD not in self.interactions.labels:
            raise RuntimeError(
                f"{func.__name__} requires an electric field. Add the "
                f"'{efieldgrad.LABEL_EFIELD_GRAD}' interaction to the "
                "Calculator."
            )
        return func(self, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efg_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    @wraps(func)
    def wrapper(
        self: Calculator,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        efg = self.interactions.get_interaction(efieldgrad.LABEL_EFIELD_GRAD)
        if not efg.field_grad.requires_grad:
            raise RuntimeError("Field gradient tensor needs ``requires_grad=True``.")
        return func(self, positions, chrg, spin, **kwargs)

    return wrapper


def _numerical(nograd: bool = False, noprint: bool = False) -> Callable[[F], F]:
    """
    Decorator for numerical differentiation.
    Pass ``True`` to turns off gradient tracking for the function.
    """

    class NoOpContext:
        def __enter__(self):
            pass

        def __exit__(self, *_, **__):
            pass

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print_context = (
                OutputHandler.with_verbosity(0) if noprint is True else NoOpContext()
            )
            grad_context = torch.no_grad() if nograd is True else NoOpContext()

            with print_context, grad_context:
                result = func(*args, **kwargs)

            return result

        return cast(F, wrapper)

    return decorator


def numerical(func: F) -> F:
    """
    Decorator for numerical differentiation. Turns off gradient tracking.

    .. warning::

        Since this decorator turns off gradient tracking for the function, a
        possible ``requires_grad=True`` will be lost because the corresponding
        tensor is updated within the numerical differentiation.
        This happens in any electric field related derivatives. If you want to
        carry out a subsequent calculation with ``requires_grad=True``, you have
        to update the electric field tensor manually with:

        .. code-block:: python

            field_tensor.requires_grad_(True)
            calc.interactions.update_efield(field=field_tensor)
    """
    return _numerical(nograd=True)(func)


def cache(func: F) -> F:
    """
    Decorator to cache the results of a function.

    .. warning::

        This decorator must always be the innermost decorator.
    """

    @wraps(func)
    def wrapper(self: Calculator, *args, **kwargs):
        cache_key: str = func.__name__
        key = cache_key.replace("_numerical", "").replace("_analytical", "")

        hashed_key = ""

        all_args = args + tuple(kwargs.values())
        for i, arg in enumerate(all_args):
            sep = "_" if i > 0 else ""
            if isinstance(arg, Tensor):
                hashed_key += f"{sep}{tensor_id(arg)}"
            else:
                hashed_key += f"{sep}{arg}"

        full_key = key + ":" + hashed_key

        # Check if the result is already in the cache in three steps:
        # 1. Are we allowed to use the cache?
        # 2. Is the key even in the cache?
        # 3. Is the cache result calculated with the same inputs?
        if self.opts.cache.enabled is True:
            if key in self.cache:
                if self.cache.get_cache_key(key) is not None:
                    if self.cache.get_cache_key(key) == full_key:
                        logger.debug(f"{cache_key.title()}: Using cached result.")
                        return self.cache[key]

        # Execute the function and store the result in the cache
        result = func(self, *args, **kwargs)
        self.cache[key] = result

        # Also store the "hashkey"
        self.cache.set_cache_key(key, full_key)
        return result

    return cast(F, wrapper)
