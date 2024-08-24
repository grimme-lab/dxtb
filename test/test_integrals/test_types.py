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
Test overlap build from integral container.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, IndexHelper, labels
from dxtb.integrals.factories import new_dipint, new_hcore, new_quadint

numbers = torch.tensor([14, 1, 1, 1, 1])
positions = torch.tensor(
    [
        [+0.00000000000000, -0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ]
)


def test_fail() -> None:
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    with pytest.raises(ValueError):
        par1 = GFN1_XTB.model_copy(deep=True)
        assert par1.meta is not None

        par1.meta.name = "fail"
        new_hcore(numbers, par1, ihelp)


def test_dipole_fail() -> None:
    i = new_dipint(labels.INTDRIVER_LIBCINT)

    with pytest.raises(RuntimeError):
        fake_ovlp = torch.eye(3, dtype=torch.float64)
        i.shift_r0_rj(fake_ovlp, positions)


def test_quadrupole_fail() -> None:
    i = new_quadint(labels.INTDRIVER_LIBCINT)

    with pytest.raises(RuntimeError):
        fake_ovlp = torch.eye(3, dtype=torch.float64)
        fake_r0 = torch.zeros(3, dtype=torch.float64)
        i.shift_r0r0_rjrj(fake_r0, fake_ovlp, positions)

    with pytest.raises(RuntimeError):
        i._matrix = torch.eye(3, dtype=torch.float64)
        i.traceless()
