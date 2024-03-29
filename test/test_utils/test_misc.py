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

from dxtb.exceptions import SCFConvergenceError
from dxtb.utils import is_int_list, is_str_list, set_jit_enabled


def test_lists() -> None:
    strlist = ["a", "b"]
    assert is_str_list(strlist)

    intlist = [1, 2]
    assert is_int_list(intlist)

    assert not is_str_list(intlist)
    assert not is_str_list(strlist + intlist)
    assert not is_int_list(strlist)
    assert not is_int_list(strlist + intlist)


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
