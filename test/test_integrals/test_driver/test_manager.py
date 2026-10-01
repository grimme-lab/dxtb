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

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.constants.labels import (
    INTDRIVER_AUTOGRAD,
    INTDRIVER_LEGACY,
    INTDRIVER_LIBCINT,
)
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint import IntDriverLibcint
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.driver.pytorch import (
    DipolePytorch,
    IntDriverPytorch,
    IntDriverPytorchLegacy,
    OverlapPytorch,
    QuadrupolePytorch,
)
from dxtb._src.typing import DD

from ...conftest import DEVICE


def test_fail() -> None:
    mgr = DriverManager(-99)

    with pytest.raises(RuntimeError):
        _ = mgr.driver

    with pytest.raises(ValueError):
        numbers = torch.tensor([1, 2], device=DEVICE)
        mgr.create_driver(
            numbers, GFN1_XTB, IndexHelper.from_numbers(numbers, GFN1_XTB)
        )


def single(name: int, dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.zeros((2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    mgr = DriverManager(name, force_cpu_for_libcint=force_cpu_for_libcint, **dd)
    mgr.create_driver(numbers, GFN1_XTB, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    mgr.setup_driver(positions)
    if name == INTDRIVER_AUTOGRAD:
        assert isinstance(mgr.driver, IntDriverPytorch)
    elif name == INTDRIVER_LIBCINT:
        assert isinstance(mgr.driver, IntDriverLibcint)
    elif name == INTDRIVER_LEGACY:
        assert isinstance(mgr.driver, IntDriverPytorchLegacy)

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
    single(INTDRIVER_AUTOGRAD, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_pytorch_legacy_single(
    dtype: torch.dtype, force_cpu_for_libcint: bool
) -> None:
    """Regression test: DriverManager previously had no dispatch branch for
    INTDRIVER_LEGACY and raised `ValueError: Unknown integral driver '3'`."""
    single(INTDRIVER_LEGACY, dtype, force_cpu_for_libcint)


def batch(name: int, dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = torch.zeros((2, 2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    mgr = DriverManager(name, force_cpu_for_libcint=force_cpu_for_libcint, **dd)
    mgr.create_driver(numbers, GFN1_XTB, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    mgr.setup_driver(positions)
    if name == INTDRIVER_AUTOGRAD:
        assert isinstance(mgr.driver, IntDriverPytorch)
    elif name == INTDRIVER_LIBCINT:
        assert isinstance(mgr.driver, IntDriverLibcint)
    elif name == INTDRIVER_LEGACY:
        assert isinstance(mgr.driver, IntDriverPytorchLegacy)

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
    batch(INTDRIVER_AUTOGRAD, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_pytorch_legacy_batch(
    dtype: torch.dtype, force_cpu_for_libcint: bool
) -> None:
    batch(INTDRIVER_LEGACY, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("kind", ["overlap", "quadrupole"])
def test_pytorch_driver_rebuilds_integrals_for_new_positions(kind: str) -> None:
    """After the positions change (or the driver is invalidated), the next
    ``setup_driver`` + ``build`` must give the integrals of the new geometry,
    identical to a freshly created driver."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([3, 1, 8], device=DEVICE)
    pos_a = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [1.5, 0.0, -1.0]], **dd
    )
    shift = torch.tensor(
        [[0.0, 0.1, 0.0], [0.2, 0.0, 0.1], [0.0, 0.0, -0.3]], **dd
    )
    pos_b = pos_a + shift

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    def manager() -> DriverManager:
        mgr = DriverManager(INTDRIVER_AUTOGRAD, algorithm="os", **dd)
        mgr.create_driver(numbers, GFN1_XTB, ihelp)
        return mgr

    def build(mgr: DriverManager) -> torch.Tensor:
        cls = {
            "overlap": OverlapPytorch,
            "dipole": DipolePytorch,
            "quadrupole": QuadrupolePytorch,
        }
        return cls[kind](**dd).build(mgr.driver)

    fresh = manager()
    fresh.setup_driver(pos_b)
    expected = build(fresh)

    mgr = manager()
    mgr.setup_driver(pos_a)
    first = build(mgr)
    assert mgr.driver.is_latest(pos_b) is False

    # moved positions: setup again and rebuild
    mgr.setup_driver(pos_b)
    moved = build(mgr)
    assert not torch.allclose(first, moved, atol=1e-6)
    assert torch.allclose(moved, expected, atol=1e-13, rtol=0.0)

    # explicit invalidation, then back to the first geometry
    mgr.invalidate_driver()
    mgr.setup_driver(pos_a)
    assert torch.allclose(build(mgr), first, atol=1e-13, rtol=0.0)
