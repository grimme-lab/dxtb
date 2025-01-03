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
Test wrappers for integrals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Param
from dxtb._src.exlibs.available import has_libcint
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


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
def test_fail(par: Param) -> None:
    with pytest.raises(ValueError):
        _par = par.model_copy(deep=True)
        _par.meta = None
        wrappers.hcore(numbers, positions, _par)

    with pytest.raises(ValueError):
        _par = par.model_copy(deep=True)
        assert _par.meta is not None

        _par.meta.name = None
        wrappers.hcore(numbers, positions, _par)

    with pytest.raises(ValueError):
        _par = par.model_copy(deep=True)
        assert _par.meta is not None

        _par.meta.name = "fail"
        wrappers.hcore(numbers, positions, _par)

    with pytest.raises(ValueError):
        # pylint: disable=import-outside-toplevel
        from dxtb._src.integral.wrappers import _integral

        _integral("fail", numbers, positions, _par)  # type: ignore


def test_h0_gfn1() -> None:
    h0 = wrappers.hcore(numbers, positions, GFN1_XTB)
    assert h0.shape == (17, 17)

    h0 = wrappers.hcore(numbers, positions, GFN1_XTB, cn=None)
    assert h0.shape == (17, 17)


def test_h0_gfn2() -> None:
    h0 = wrappers.hcore(numbers, positions, GFN2_XTB)
    assert h0.shape == (13, 13)

    h0 = wrappers.hcore(numbers, positions, GFN2_XTB, cn=None)
    assert h0.shape == (13, 13)


def test_overlap() -> None:
    s = wrappers.overlap(numbers, positions, GFN1_XTB)
    assert s.shape == (17, 17)

    s = wrappers.overlap(numbers, positions, GFN2_XTB)
    assert s.shape == (13, 13)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_dipole() -> None:
    s = wrappers.dipint(numbers, positions, GFN1_XTB)
    assert s.shape == (3, 17, 17)

    s = wrappers.dipint(numbers, positions, GFN2_XTB)
    assert s.shape == (3, 13, 13)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_quad() -> None:
    s = wrappers.quadint(numbers, positions, GFN1_XTB)
    assert s.shape == (9, 17, 17)

    s = wrappers.quadint(numbers, positions, GFN2_XTB)
    assert s.shape == (9, 13, 13)
