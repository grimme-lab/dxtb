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
Test Calculator setup.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.exceptions import DtypeError

from dxtb.param import GFN1_XTB as par
from dxtb.timing import timer
from dxtb.xtb import Calculator


def test_fail() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1], dtype=torch.double)

    with pytest.raises(DtypeError):
        Calculator(numbers, par, opts={"vebosity": 0})

    # because of the exception, the timer for the setup is never stopped
    timer.reset()


def run_asserts(c: Calculator, dtype: torch.dtype) -> None:
    assert c.dtype == dtype
    assert c.classicals.dtype == dtype
    assert c.interactions.dtype == dtype
    assert c.opts.dtype == dtype

    assert c.integrals.dtype == dtype
    assert c.integrals.hcore is not None
    assert c.integrals.hcore.dtype == dtype
    assert c.integrals.overlap is not None
    assert c.integrals.overlap.dtype == dtype

    assert c.integrals.matrices.dtype == dtype

    assert c.integrals.driver.dtype == dtype


def test_change_type() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1])
    calc = Calculator(numbers, par, dtype=torch.double)

    calc = calc.type(torch.float32)
    run_asserts(calc, torch.float32)

    timer.reset()


def test_change_type_after_energy() -> None:
    dtype = torch.float64

    numbers = torch.tensor([1, 1])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

    calc_64 = Calculator(numbers, par, dtype=dtype, opts={"verbosity": 0})
    calc_64.energy(pos)

    run_asserts(calc_64, dtype)

    # extra asserts on initialized vars
    assert calc_64.integrals.driver.basis.dtype == dtype
    assert calc_64.integrals.driver.basis.ngauss.dtype == torch.uint8
    assert calc_64.integrals.driver.basis.pqn.dtype == torch.uint8
    assert calc_64.integrals.driver.basis.slater.dtype == dtype

    assert calc_64.integrals.matrices.hcore is not None
    assert calc_64.integrals.matrices.hcore.dtype == dtype
    assert calc_64.integrals.matrices.overlap is not None
    assert calc_64.integrals.matrices.overlap.dtype == dtype

    ############################################################################

    calc_32 = calc_64.type(torch.float32)
    run_asserts(calc_32, torch.float32)

    timer.reset()
