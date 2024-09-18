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
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint import IntDriverLibcint
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.driver.pytorch import IntDriverPytorch
from dxtb._src.typing import DD

from ...conftest import DEVICE


def test_fail() -> None:
    mgr = DriverManager(-99)

    with pytest.raises(RuntimeError):
        _ = mgr.driver

    with pytest.raises(ValueError):
        numbers = torch.tensor([1, 2], device=DEVICE)
        mgr.create_driver(numbers, par, IndexHelper.from_numbers(numbers, par))


def single(name: int, dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.zeros((2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    mgr = DriverManager(name, force_cpu_for_libcint=force_cpu_for_libcint, **dd)
    mgr.create_driver(numbers, par, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    mgr.setup_driver(positions)
    if name == INTDRIVER_ANALYTICAL:
        assert isinstance(mgr.driver, IntDriverPytorch)
    elif name == INTDRIVER_LIBCINT:
        assert isinstance(mgr.driver, IntDriverLibcint)

    assert mgr.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert mgr.driver.is_latest(positions) is False


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_libcint_single(
    dtype: torch.dtype, force_cpu_for_libcint: bool
) -> None:
    single(INTDRIVER_LIBCINT, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_pytorch_single(
    dtype: torch.dtype, force_cpu_for_libcint: bool
) -> None:
    single(INTDRIVER_ANALYTICAL, dtype, force_cpu_for_libcint)


def batch(name: int, dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = torch.zeros((2, 2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    mgr = DriverManager(name, force_cpu_for_libcint=force_cpu_for_libcint, **dd)
    mgr.create_driver(numbers, par, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    mgr.setup_driver(positions)
    if name == INTDRIVER_ANALYTICAL:
        assert isinstance(mgr.driver, IntDriverPytorch)
    elif name == INTDRIVER_LIBCINT:
        assert isinstance(mgr.driver, IntDriverLibcint)

    assert mgr.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert mgr.driver.is_latest(positions) is False


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_libcint_batch(dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    batch(INTDRIVER_LIBCINT, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_pytorch_batch(dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    batch(INTDRIVER_ANALYTICAL, dtype, force_cpu_for_libcint)
