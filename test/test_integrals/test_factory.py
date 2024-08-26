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
Test factories for integral classes.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, labels
from dxtb._src.integral import factory
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
from dxtb._src.xtb.gfn2 import GFN2Hamiltonian
from dxtb.integrals import factories, types

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
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    with pytest.raises(ValueError):
        par1 = GFN1_XTB.model_copy(deep=True)
        par1.meta = None
        factories.new_hcore(numbers, par1, ihelp)

    with pytest.raises(ValueError):
        par1 = GFN1_XTB.model_copy(deep=True)
        assert par1.meta is not None

        par1.meta.name = None
        factories.new_hcore(numbers, par1, ihelp)

    with pytest.raises(ValueError):
        par1 = GFN1_XTB.model_copy(deep=True)
        assert par1.meta is not None

        par1.meta.name = "fail"
        factories.new_hcore(numbers, par1, ihelp)


def test_hcore() -> None:
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    h0_gfn1 = factory.new_hcore(numbers, GFN1_XTB, ihelp)
    assert isinstance(h0_gfn1, GFN1Hamiltonian)

    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)
    h0_gfn2 = factory.new_hcore(numbers, GFN2_XTB, ihelp)
    assert isinstance(h0_gfn2, GFN2Hamiltonian)


def test_hcore_gfn1() -> None:
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    h0 = factory.new_hcore_gfn1(numbers, ihelp)
    assert isinstance(h0, GFN1Hamiltonian)

    h0 = factory.new_hcore_gfn1(numbers, ihelp, GFN1_XTB)
    assert isinstance(h0, GFN1Hamiltonian)


def test_hcore_gfn2() -> None:
    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)

    h0 = factory.new_hcore_gfn2(numbers, ihelp)
    assert isinstance(h0, GFN2Hamiltonian)

    h0 = factory.new_hcore_gfn2(numbers, ihelp, GFN2_XTB)
    assert isinstance(h0, GFN2Hamiltonian)


################################################################################


def test_overlap_fail() -> None:
    with pytest.raises(ValueError):
        factory.new_overlap(-1)


def test_overlap() -> None:
    cls = factory.new_overlap(labels.INTDRIVER_LIBCINT)
    assert isinstance(cls, types.OverlapIntegral)

    cls = factory.new_overlap(labels.INTDRIVER_ANALYTICAL)
    assert isinstance(cls, types.OverlapIntegral)


def test_overlap_libcint() -> None:
    cls = factory.new_overlap_libcint()
    assert isinstance(cls, types.OverlapIntegral)
    assert cls.device == torch.device("cpu")

    cls = factory.new_overlap_libcint(force_cpu_for_libcint=True)
    assert isinstance(cls, types.OverlapIntegral)
    assert cls.device == torch.device("cpu")


def test_overlap_pytorch() -> None:
    cls = factory.new_overlap_pytorch()
    assert isinstance(cls, types.OverlapIntegral)


################################################################################


def test_dipint_fail() -> None:
    with pytest.raises(ValueError):
        factory.new_dipint(-1)


def test_dipint() -> None:
    cls = factory.new_dipint(labels.INTDRIVER_LIBCINT)
    assert isinstance(cls, types.DipoleIntegral)

    with pytest.raises(NotImplementedError):
        cls = factory.new_dipint(labels.INTDRIVER_ANALYTICAL)
        assert isinstance(cls, types.DipoleIntegral)


def test_dipint_libcint() -> None:
    cls = factory.new_dipint_libcint()
    assert isinstance(cls, types.DipoleIntegral)
    assert cls.device == torch.device("cpu")

    cls = factory.new_dipint_libcint(force_cpu_for_libcint=True)
    assert isinstance(cls, types.DipoleIntegral)
    assert cls.device == torch.device("cpu")


def test_dipint_pytorch() -> None:
    with pytest.raises(NotImplementedError):
        cls = factory.new_dipint_pytorch()
        assert isinstance(cls, types.DipoleIntegral)


################################################################################


def test_quadint_fail() -> None:
    with pytest.raises(ValueError):
        factory.new_quadint(-1)


def test_quadint() -> None:
    cls = factory.new_quadint(labels.INTDRIVER_LIBCINT)
    assert isinstance(cls, types.QuadrupoleIntegral)

    with pytest.raises(NotImplementedError):
        cls = factory.new_quadint(labels.INTDRIVER_ANALYTICAL)
        assert isinstance(cls, types.QuadrupoleIntegral)


def test_quadint_libcint() -> None:
    cls = factory.new_quadint_libcint()
    assert isinstance(cls, types.QuadrupoleIntegral)
    assert cls.device == torch.device("cpu")

    cls = factory.new_quadint_libcint(force_cpu_for_libcint=True)
    assert isinstance(cls, types.QuadrupoleIntegral)
    assert cls.device == torch.device("cpu")


def test_quadint_pytorch() -> None:
    with pytest.raises(NotImplementedError):
        cls = factory.new_quadint_pytorch()
        assert isinstance(cls, types.QuadrupoleIntegral)
