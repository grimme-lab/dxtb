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
Test Calculator dtype and device consistency.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.exceptions import DeviceError, DtypeError

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.typing import MockTensor, Tensor


def test_fail_dtype() -> None:
    numbers = torch.tensor([3, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    charge = torch.tensor(0.0)
    spin = torch.tensor(0.0)

    calc = Calculator(numbers, par, opts={"verbosity": 0})

    # same dtype works
    e = calc.get_energy(positions, charge, spin)
    assert isinstance(e, Tensor)

    with pytest.raises(DtypeError):
        calc.get_energy(positions.type(torch.double), charge, spin)

    with pytest.raises(DtypeError):
        calc.get_energy(positions, charge.type(torch.double), spin)

    with pytest.raises(DtypeError):
        calc.get_energy(positions, charge, spin.type(torch.double))


def test_fail_device() -> None:
    numbers = torch.tensor([3, 1])

    _positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _charge = torch.tensor(0.0)
    _spin = torch.tensor(0.0)

    calc = Calculator(numbers, par, opts={"verbosity": 0}, dtype=torch.float)

    # same device works
    e = calc.get_energy(_positions, _charge, _spin)
    assert isinstance(e, Tensor)

    with pytest.raises(DeviceError):
        positions = MockTensor(_positions)
        positions.device = torch.device("cuda")
        calc.get_energy(positions, _charge, _spin)

    with pytest.raises(DeviceError):
        charge = MockTensor(_charge)
        charge.device = torch.device("cuda")
        calc.get_energy(_positions, charge, _spin)

    with pytest.raises(DeviceError):
        spin = MockTensor(_spin)
        spin.device = torch.device("cuda")
        calc.get_energy(_positions, _charge, spin)
