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
Test Calculator usage.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB
from dxtb.calculators import AnalyticalCalculator, GFN1Calculator
from dxtb.typing import DD

DEVICE = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    opts = {"use_cache": True}
    calc = GFN1Calculator(numbers, opts=opts, **dd)
    assert calc.ncalcs == 0

    energy = calc.get_energy(positions)
    assert calc.ncalcs == 1
    assert isinstance(energy, torch.Tensor)

    # cache is used
    energy = calc.get_energy(positions)
    assert calc.ncalcs == 1
    assert isinstance(energy, torch.Tensor)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_forces(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    positions.requires_grad_(True)

    opts = {"use_cache": True}
    calc = AnalyticalCalculator(numbers, GFN1_XTB, opts=opts, **dd)
    assert calc.ncalcs == 0

    prop = calc.get_forces(positions)
    assert calc.ncalcs == 1
    assert isinstance(prop, torch.Tensor)

    # cache is used
    prop = calc.get_forces(positions)
    assert calc.ncalcs == 1
    assert isinstance(prop, torch.Tensor)
