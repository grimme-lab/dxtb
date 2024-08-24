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
Test the integral driver manager.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.constants.labels import INTDRIVER_ANALYTICAL, INTDRIVER_LIBCINT
from dxtb._src.integral.driver.libcint import IntDriverLibcint
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.driver.pytorch import IntDriverPytorch
from dxtb._src.typing import DD

from ...conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_single(dtype: torch.dtype, force_cpu_for_libcint: bool):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.zeros((2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    mgr_py = DriverManager(
        INTDRIVER_ANALYTICAL, force_cpu_for_libcint=force_cpu_for_libcint, **dd
    )
    mgr_py.create_driver(numbers, par, ihelp)

    mgr_lc = DriverManager(
        INTDRIVER_LIBCINT, force_cpu_for_libcint=force_cpu_for_libcint, **dd
    )
    mgr_lc.create_driver(numbers, par, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    mgr_py.setup_driver(positions)
    assert isinstance(mgr_py.driver, IntDriverPytorch)
    mgr_lc.setup_driver(positions)
    assert isinstance(mgr_lc.driver, IntDriverLibcint)

    assert mgr_py.driver.is_latest(positions) is True
    assert mgr_lc.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert mgr_py.driver.is_latest(positions) is False
    assert mgr_lc.driver.is_latest(positions) is False


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_batch(dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = torch.zeros((2, 2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    mgr_py = DriverManager(
        INTDRIVER_ANALYTICAL, force_cpu_for_libcint=force_cpu_for_libcint, **dd
    )
    mgr_py.create_driver(numbers, par, ihelp)

    mgr_lc = DriverManager(
        INTDRIVER_LIBCINT, force_cpu_for_libcint=force_cpu_for_libcint, **dd
    )
    mgr_lc.create_driver(numbers, par, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    mgr_py.setup_driver(positions)
    assert isinstance(mgr_py.driver, IntDriverPytorch)
    mgr_lc.setup_driver(positions)
    assert isinstance(mgr_lc.driver, IntDriverLibcint)

    assert mgr_py.driver.is_latest(positions) is True
    assert mgr_lc.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert mgr_py.driver.is_latest(positions) is False
    assert mgr_lc.driver.is_latest(positions) is False
