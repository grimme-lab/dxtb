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

import torch

from dxtb._src.typing import TYPE_CHECKING, Any, Tensor, TypeGuard, TypeVar

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "get_all_slots",
    "is_str_list",
    "is_int_list",
    "is_basis_list",
    "convert_float_tensor",
    "set_jit_enabled",
]


T = TypeVar("T")


def get_all_slots(cls):
    # cls.__class__.__mro__ = (<class 'object'>, <class 'TensorLike'>,
    # <class 'BaseResult'>, <class 'VibResult'>)

    # skip the "object" parent and the current class itself
    parents = [
        p
        for p in cls.__class__.__mro__
        if p.__name__ not in ("object", cls.__class__.__name__)
    ]

    #  and the hidden slots "__" and the "unit" slots
    parents_slots: list[str] = [
        s for p in parents for s in p.__slots__ if "__" not in s
    ]

    # add the slots of the current class (after parents for ordering)
    return parents_slots + cls.__slots__


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
        ``True`` if all objects are strings, ``False`` otherwise.
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
        ``True`` if all objects are integers, ``False`` otherwise.
    """
    if not isinstance(x, list):
        return False
    return all(isinstance(i, int) for i in x)


def is_basis_list(x: Any) -> TypeGuard[list[libcint.AtomCGTOBasis]]:
    """
    Determines whether all objects in the list are `AtomCGTOBasis`.

    Parameters
    ----------
    x : list[Any]
        List to check.

    Returns
    -------
    TypeGuard[list[AtomCGTOBasis]]
        ``True`` if all objects are `AtomCGTOBasis`, ``False`` otherwise.
    """
    if not isinstance(x, list):
        return False

    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    return all(isinstance(i, libcint.AtomCGTOBasis) for i in x)


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
