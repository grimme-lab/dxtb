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
Test optional cached quantities.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.typing import DD, Tensor
from dxtb.calculators import GFN1Calculator
from ...conftest import DEVICE


opts = {"cache_enabled": True, "verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_density(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    calc = GFN1Calculator(numbers, opts=opts, **dd)
    assert calc._ncalcs == 0

    # enable density caching
    calc.opts.cache.store.density = True

    energy = calc.get_energy(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    density = calc.get_density(positions)
    assert calc._ncalcs == 1
    assert isinstance(density, Tensor)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bond_orders(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    calc = GFN1Calculator(numbers, opts=opts, **dd)
    assert calc._ncalcs == 0

    # enable caching of properties required for bond orders
    calc.opts.cache.store.density = True
    calc.opts.cache.store.overlap = True

    energy = calc.get_bond_orders(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)
