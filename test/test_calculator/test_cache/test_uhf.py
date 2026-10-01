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
Test Calculator cache for unrestricted (UHF) calculations.
"""

# pylint: disable=protected-access
from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.typing import DD
from dxtb.components.spin import new_spin_polarization

from ...conftest import DEVICE

opts = {"cache_enabled": True, "verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_nspin(dtype: torch.dtype) -> None:
    """Restricted and unrestricted results are cached separately."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], **dd)

    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    calc.opts.cache.store.density = True

    density = calc.get_density(positions)
    assert calc._ncalcs == 1
    assert density.shape == (6, 6)

    # cache is used
    calc.get_density(positions)
    assert calc._ncalcs == 1

    # switching to UHF invalidates the cache
    calc.opts.scf.uhf_mode = True
    density = calc.get_density(positions)
    assert calc._ncalcs == 2
    assert density.shape == (2, 6, 6)

    # adding spin polarization requires UHF (same number of channels)
    calc.opts.scf.uhf_mode = False
    calc.interactions.components.append(new_spin_polarization(numbers, **dd))
    density = calc.get_density(positions)
    assert density.shape == (2, 6, 6)

    # spin is part of the cache key
    calc.get_density(positions, spin=2)
    assert calc._ncalcs == 3
