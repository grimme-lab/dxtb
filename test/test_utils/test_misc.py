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
Test additional utility functions.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing.exceptions import SCFConvergenceError
from dxtb._src.utils import (
    is_basis_list,
    is_float,
    is_float_list,
    is_int_list,
    is_integer,
    is_numeric,
    is_str_list,
    set_jit_enabled,
)


def test_lists_string() -> None:
    """Check if the input is a list of strings."""
    assert is_str_list(["a", "b", "c"]) is True
    assert is_str_list(["a", 1, "c"]) is False
    assert is_str_list([]) is True
    assert is_str_list(["a", ""]) is True
    assert is_str_list([False]) is False
    assert is_str_list(123) is False  # type: ignore
    assert is_str_list(None) is False  # type: ignore


def test_list_integer() -> None:
    """Test if the input is a list of integers."""
    assert is_int_list([1, 2, 3]) is True
    assert is_int_list([1, "a", 3]) is False
    assert is_int_list([]) is True
    assert is_int_list([1, -1, 0]) is True
    assert is_int_list("123") is False
    assert is_int_list([False]) is False
    assert is_int_list(None) is False


def test_list_float() -> None:
    """Test if the input is a list of floats."""
    assert is_float_list([1.0, 2.0, 3.0]) is True
    assert is_float_list([1.0, "a", 3.0]) is False
    assert is_float_list([]) is True
    assert is_float_list([1.0, -1.0, 0.0]) is True
    assert is_float_list("123") is False
    assert is_float_list(None) is False


def test_numeric() -> None:
    """Test if the input is numeric."""
    assert is_numeric(1) is True
    assert is_numeric(1.0) is True
    assert is_numeric("a") is False
    assert is_numeric([]) is False
    assert is_numeric([1, 2]) is False
    assert is_numeric(None) is False
    assert is_integer(False) is False


def test_integer() -> None:
    """Test if the input is an integer."""
    assert is_integer(1) is True
    assert is_integer(1.0) is False
    assert is_integer("a") is False
    assert is_integer([]) is False
    assert is_integer([1, 2]) is False
    assert is_integer(None) is False
    assert is_integer(False) is False


def test_float() -> None:
    """Test if the input is a float."""
    assert is_float(1.0) is True
    assert is_float(1) is False
    assert is_float("a") is False
    assert is_float([]) is False
    assert is_float([1, 2]) is False
    assert is_float(None) is False
    assert is_float(False) is False


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_is_basis_list() -> None:
    """Test if the input is a list of AtomCGTOBasis."""
    from dxtb._src.exlibs import libcint  # type: ignore

    basis = libcint.AtomCGTOBasis(
        1,
        [libcint.CGTOBasis(1, torch.tensor([1.0]), torch.tensor([1.0]))],
        torch.tensor([0.0, 0.0, 0.0]),
    )

    assert is_basis_list([basis, basis]) is True
    assert is_basis_list([basis, "a"]) is False
    assert is_basis_list([]) is True
    assert is_basis_list("basis") is False
    assert is_basis_list(None) is False


def test_jit_settings() -> None:
    """Test the JIT settings."""
    # save current state
    state = torch.jit._state._enabled.enabled  # type: ignore

    set_jit_enabled(True)
    assert torch.jit._state._enabled.enabled  # type: ignore

    set_jit_enabled(False)
    assert not torch.jit._state._enabled.enabled  # type: ignore

    # restore initial state
    set_jit_enabled(state)


def test_exceptions() -> None:
    """Test exceptions."""
    msg = "The error message."
    with pytest.raises(SCFConvergenceError, match=msg):
        raise SCFConvergenceError(msg)
