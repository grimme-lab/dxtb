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
"""Removed SCF modes raise an error that names the replacement."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.constants import labels


@pytest.mark.parametrize(
    "mode", list(labels.SCF_MODE_REMOVED_STRS) + ["NonPure"]
)
def test_removed_string(mode: str) -> None:
    """Removed mode names raise a ValueError that names `implicit`."""
    with pytest.raises(ValueError, match="removed.*'implicit'"):
        Calculator(torch.tensor([1, 1]), GFN1_XTB, opts={"scf_mode": mode})


def test_removed_int() -> None:
    """The removed integer code 2 raises a ValueError naming `implicit`."""
    with pytest.raises(ValueError, match="code 2.*removed.*'implicit'"):
        Calculator(torch.tensor([1, 1]), GFN1_XTB, opts={"scf_mode": 2})


@pytest.mark.parametrize("mode", ["implicit", "default", "full", 0, 1, 3])
def test_remaining_modes_accepted(mode: str | int) -> None:
    """The remaining modes are still accepted."""
    Calculator(torch.tensor([1, 1]), GFN1_XTB, opts={"scf_mode": mode})


@pytest.mark.parametrize(
    "name", ["SCF_MODE_IMPLICIT_NON_PURE", "SCF_MODE_IMPLICIT_NON_PURE_STRS"]
)
def test_removed_constant_is_kept(name: str) -> None:
    """The old public constants still exist and lead to the removal error."""
    from dxtb import labels as public

    value = getattr(public, name)
    modes = value if isinstance(value, tuple) else (value,)
    for mode in modes:
        with pytest.raises(ValueError, match="removed.*'implicit'"):
            Calculator(torch.tensor([1, 1]), GFN1_XTB, opts={"scf_mode": mode})
