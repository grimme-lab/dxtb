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

from dxtb import GFN1_XTB, GFN2_XTB, Param
from dxtb.integrals import wrappers

numbers = torch.tensor([14, 1, 1, 1, 1])
positions = torch.tensor(
    [
        [+0.00000000000000, +0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ]
)


def test_fail() -> None:
    with pytest.raises(TypeError):
        par1 = GFN1_XTB.model_copy(deep=True)
        par1.meta = None
        wrappers.hcore(numbers, positions, par1)

    with pytest.raises(TypeError):
        par1 = GFN1_XTB.model_copy(deep=True)
        assert par1.meta is not None

        par1.meta.name = None
        wrappers.hcore(numbers, positions, par1)

    with pytest.raises(ValueError):
        par1 = GFN1_XTB.model_copy(deep=True)
        assert par1.meta is not None

        par1.meta.name = "fail"
        wrappers.hcore(numbers, positions, par1)

    with pytest.raises(ValueError):
        # pylint: disable=import-outside-toplevel
        from dxtb._src.integral.wrappers import _integral

        _integral("fail", numbers, positions, par1)  # type: ignore


@pytest.mark.parametrize("par", [GFN1_XTB])
def test_h0_gfn1(par: Param) -> None:
    h0 = wrappers.hcore(numbers, positions, par)
    assert h0.shape == (17, 17)

    h0 = wrappers.hcore(numbers, positions, par, cn=None)
    assert h0.shape == (17, 17)


@pytest.mark.parametrize("par", [GFN2_XTB])
def test_h0_gfn2(par: Param) -> None:
    with pytest.raises(NotImplementedError):
        wrappers.hcore(numbers, positions, par)


def test_overlap() -> None:
    s = wrappers.overlap(numbers, positions, GFN1_XTB)
    assert s.shape == (17, 17)


def test_dipole() -> None:
    s = wrappers.dipint(numbers, positions, GFN1_XTB)
    assert s.shape == (3, 17, 17)


def test_quad() -> None:
    s = wrappers.quadint(numbers, positions, GFN1_XTB)
    assert s.shape == (9, 17, 17)
