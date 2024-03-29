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

from dxtb import integral as ints
from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.constants.labels import INTDRIVER_ANALYTICAL, INTDRIVER_LIBCINT
from dxtb.param import GFN1_XTB as par

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=device)
    positions = torch.zeros((2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    ipy = ints.Integrals(numbers, par, ihelp, driver=INTDRIVER_ANALYTICAL, **dd)
    ilc = ints.Integrals(numbers, par, ihelp, driver=INTDRIVER_LIBCINT, **dd)

    ipy.setup_driver(positions)
    assert isinstance(ipy.driver, ints.driver.IntDriverPytorch)
    ilc.setup_driver(positions)
    assert isinstance(ilc.driver, ints.driver.IntDriverLibcint)

    assert ipy.driver.is_latest(positions) is True
    assert ilc.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert ipy.driver.is_latest(positions) is False
    assert ilc.driver.is_latest(positions) is False


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([[3, 1], [1, 0]], device=device)
    positions = torch.zeros((2, 2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    ipy = ints.Integrals(numbers, par, ihelp, driver=INTDRIVER_ANALYTICAL, **dd)
    ilc = ints.Integrals(numbers, par, ihelp, driver=INTDRIVER_LIBCINT, **dd)

    ipy.setup_driver(positions)
    assert isinstance(ipy.driver, ints.driver.IntDriverPytorch)
    ilc.setup_driver(positions)
    assert isinstance(ilc.driver, ints.driver.IntDriverLibcint)

    assert ipy.driver.is_latest(positions) is True
    assert ilc.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert ipy.driver.is_latest(positions) is False
    assert ilc.driver.is_latest(positions) is False
