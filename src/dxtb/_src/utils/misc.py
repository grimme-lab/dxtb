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

from dxtb._src.typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "get_all_slots",
    "is_str_list",
    "is_integer",
    "is_float",
    "is_numeric",
    "is_int_list",
    "is_float_list",
    "is_basis_list",
    "set_jit_enabled",
]


def get_all_slots(cls):
    # cls.__class__.__mro__ = (<class 'object'>, <class 'TensorLike'>,
    # <class 'BaseResult'>, <class 'VibResult'>)

    # skip the "object" parent and the current class itself
    parents = [
        p
        for p in cls.__class__.__mro__
        if p.__name__ not in ("object", cls.__class__.__name__)
    ]

    # and the hidden slots "__" and the "unit" slots
    parents_slots: list[str] = [
        s for p in parents for s in p.__slots__ if "__" not in s
    ]

    # add the slots of the current class (after parents for ordering)
    return parents_slots + cls.__slots__


def is_integer(val: Any) -> TypeGuard[int]:
    """Return ``True`` if *val* is an integer (excluding booleans)."""
    # bool inherits from int!
    return isinstance(val, int) and not isinstance(val, bool)


def is_float(val: Any) -> TypeGuard[float]:
    """Return ``True`` if *val* is a float."""
    return isinstance(val, float)


def is_numeric(val: Any) -> TypeGuard[list[float | int]]:
    """Return ``True`` if *val* is either an integer or a float."""
    return is_integer(val) or is_float(val)


def is_str_list(lst: list[Any]) -> TypeGuard[list[str]]:
    """Return ``True`` if *lst* is a list of string values"""
    return isinstance(lst, list) and all(isinstance(i, str) for i in lst)


def is_int_list(lst: Any) -> TypeGuard[list[int]]:
    """Return ``True`` if *lst* is a list of integer values."""
    return isinstance(lst, list) and all(is_integer(x) for x in lst)


def is_float_list(lst: Any) -> TypeGuard[list[float]]:
    """Return ``True`` if *lst* is a list of flaot values."""
    return isinstance(lst, list) and all(is_float(x) for x in lst)


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
