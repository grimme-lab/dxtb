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
Run tests for overlap of diatomic systems.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.integral.driver.pytorch.impls.md import overlap_gto
from dxtb._src.typing.exceptions import (
    CGTOAzimuthalQuantumNumberError,
    CGTOPrimitivesError,
    CGTOPrincipalQuantumNumberError,
    CGTOQuantumNumberError,
    CGTOSlaterExponentsError,
    IntegralTransformError,
)


def test_fail_number_primitives() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOPrimitivesError):
        slater_to_gauss(torch.tensor(7), n, l, torch.tensor(1.2))


def test_fail_slater_exponent() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOSlaterExponentsError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(-1.2))


def test_fail_max_principal() -> None:
    # principal and azimuthal quantum number of 7s-orbital
    n, l = torch.tensor(7), torch.tensor(0)

    with pytest.raises(CGTOPrincipalQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital() -> None:
    # principal and azimuthal quantum number of 5h-orbital
    n, l = torch.tensor(5), torch.tensor(5)

    with pytest.raises(CGTOAzimuthalQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_quantum_number() -> None:
    # principal and azimuthal quantum number of 2f-orbital
    n, l = torch.tensor(2), torch.tensor(3)

    with pytest.raises(CGTOQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital_trafo():
    """No higher orbitals than d-orbitals allowed."""
    vec = torch.tensor([0.0, 0.0, 1.4])

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, par)
    bas = Basis(number, par, ihelp)
    alpha, coeff = bas.create_cgtos()

    j = torch.tensor(5)
    for i in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_gto(
                (torch.tensor(i), j),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )
    i = torch.tensor(5)
    for j in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_gto(
                (i, torch.tensor(j)),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )


# Device and Dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Test changing the `dtype` of the Basis class."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, par)

    bas = Basis(number, par, ihelp)
    assert bas is not None

    cls = bas.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    """Test failure upon changing `dtype` incorrectly."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, par)

    bas = Basis(number, par, ihelp)
    assert bas is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        bas.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        bas.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    """Test changing the `device` of the class."""
    device = str_to_device(device_str)

    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, par)

    bas = Basis(number, par, ihelp)
    assert bas is not None

    bas = bas.to(device)
    assert bas.device == device


def test_change_device_fail() -> None:
    """Test failure upon changing `device` incorrectly."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, par)

    bas = Basis(number, par, ihelp)
    assert bas is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        bas.device = "cpu"
