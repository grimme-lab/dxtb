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

from dxtb._src.typing.exceptions import SCFConvergenceError
from dxtb._src.utils import is_basis_list, is_int_list, is_str_list, set_jit_enabled


def test_lists() -> None:
    assert is_str_list(["a", "b", "c"]) == True
    assert is_str_list(["a", 1, "c"]) == False
    assert is_str_list([]) == True
    assert is_str_list(["a", ""]) == True
    assert is_str_list(123) == False  # type: ignore
    assert is_str_list(None) == False  # type: ignore


def test_is_int_list():
    assert is_int_list([1, 2, 3]) == True
    assert is_int_list([1, "a", 3]) == False
    assert is_int_list([]) == True
    assert is_int_list([1, -1, 0]) == True
    assert is_int_list("123") == False  # type: ignore
    assert is_int_list(None) == False  # type: ignore


def test_is_basis_list(monkeypatch):
    # Mocking the import inside the function
    from dxtb._src.exlibs.libcint import AtomCGTOBasis, CGTOBasis

    basis = AtomCGTOBasis(
        1,
        [CGTOBasis(1, torch.tensor([1.0]), torch.tensor([1.0]))],
        torch.tensor([0.0, 0.0, 0.0]),
    )

    assert is_basis_list([basis, basis]) == True
    assert is_basis_list([basis, "a"]) == False
    assert is_basis_list([]) == True
    assert is_basis_list("basis") == False
    assert is_basis_list(None) == False


def test_jit_settings() -> None:
    # save current state
    state = torch.jit._state._enabled.enabled  # type: ignore

    set_jit_enabled(True)
    assert torch.jit._state._enabled.enabled  # type: ignore

    set_jit_enabled(False)
    assert not torch.jit._state._enabled.enabled  # type: ignore

    # restore initial state
    set_jit_enabled(state)


def test_exceptions() -> None:
    msg = "The error message."
    with pytest.raises(SCFConvergenceError, match=msg):
        raise SCFConvergenceError(msg)
