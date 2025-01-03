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

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb import integrals as ints
from dxtb._src.constants.labels import INTDRIVER_ANALYTICAL, INTDRIVER_LIBCINT
from dxtb._src.integral.driver import libcint, pytorch
from dxtb._src.typing import DD
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_properties(dtype: torch.dtype):
    dd: DD = {"dtype": dtype, "device": DEVICE}

    mgr = ints.DriverManager(INTDRIVER_LIBCINT, **dd)
    i = ints.Integrals(mgr, **dd)

    assert len(i.labels) == 0
    assert len(i) == 0


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_empty(dtype: torch.dtype):
    dd: DD = {"dtype": dtype, "device": DEVICE}

    mgr = ints.DriverManager(INTDRIVER_LIBCINT, **dd)
    i = ints.Integrals(mgr, **dd)

    assert i._hcore is None
    assert i._overlap is None
    assert i._dipole is None
    assert i._quadrupole is None

    assert i.hcore is None
    assert i.overlap is None
    assert i.dipole is None
    assert i.quadrupole is None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_family(dtype: torch.dtype):
    dd: DD = {"dtype": dtype, "device": DEVICE}
    numbers = torch.tensor([1, 3], device=DEVICE)

    ihelp = IndexHelper.from_numbers(numbers, par)
    mgr = ints.DriverManager(INTDRIVER_ANALYTICAL, **dd)
    mgr.create_driver(numbers, par, ihelp)

    i = ints.Integrals(mgr, **dd)

    # make sure the checks are turned on
    assert i.run_checks is True

    with pytest.raises(RuntimeError):
        i.overlap = libcint.OverlapLibcint(**dd)
    with pytest.raises(RuntimeError):
        i.dipole = libcint.DipoleLibcint(**dd)
    with pytest.raises(RuntimeError):
        i.quadrupole = libcint.QuadrupoleLibcint(**dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_pytorch_multipole(dtype: torch.dtype):
    dd: DD = {"dtype": dtype, "device": DEVICE}
    numbers = torch.tensor([1, 3], device=DEVICE)

    ihelp = IndexHelper.from_numbers(numbers, par)
    mgr = ints.DriverManager(INTDRIVER_LIBCINT, **dd)
    mgr.create_driver(numbers, par, ihelp)

    i = ints.Integrals(mgr, **dd)

    # make sure the checks are turned on
    assert i.run_checks is True

    # incompatible driver
    with pytest.raises(RuntimeError):
        i.overlap = pytorch.OverlapPytorch(**dd)

    # multipole moments not implemented with PyTorch
    with pytest.raises(NotImplementedError):
        i.dipole = pytorch.DipolePytorch(**dd)
    with pytest.raises(NotImplementedError):
        i.quadrupole = pytorch.QuadrupolePytorch(**dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_hcore(dtype: torch.dtype):
    dd: DD = {"dtype": dtype, "device": DEVICE}
    numbers = torch.tensor([1, 3], device=DEVICE)

    ihelp = IndexHelper.from_numbers(numbers, par)
    mgr = ints.DriverManager(INTDRIVER_ANALYTICAL, **dd)
    mgr.create_driver(numbers, par, ihelp)

    i = ints.Integrals(mgr, **dd)
    i.hcore = GFN1Hamiltonian(numbers, par, ihelp, **dd)

    h = i.hcore
    assert h is not None
    assert h.matrix is None
