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
Test the PyTorch integral driver.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.integral.driver.pytorch import (
    DipolePytorch,
    OverlapPytorch,
    QuadrupolePytorch,
)
from dxtb._src.integral.driver.pytorch.driver import BaseIntDriverPytorch
from dxtb._src.typing import DD

from ...conftest import DEVICE


def test_overlap_fail() -> None:
    with pytest.raises(ValueError):
        _ = OverlapPytorch(uplo="wrong")  # type: ignore


def test_dipole_fail() -> None:
    with pytest.raises(NotImplementedError):
        _ = DipolePytorch()

    with pytest.raises(ValueError):
        _ = DipolePytorch("wrong")  # type: ignore


def test_quadrupole_fail() -> None:
    with pytest.raises(NotImplementedError):
        _ = QuadrupolePytorch()

    with pytest.raises(ValueError):
        _ = QuadrupolePytorch("wrong")  # type: ignore


##############################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype):
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.zeros((2, 3), **dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    drv = BaseIntDriverPytorch(numbers, GFN1_XTB, ihelp, **dd)
    drv.setup(positions)

    assert drv._basis is not None
    assert drv._positions is not None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_mode_fail(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = torch.zeros((2, 2, 3), **dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # set to invalid value
    ihelp.batch_mode = -99

    drv = BaseIntDriverPytorch(numbers, GFN1_XTB, ihelp, **dd)

    with pytest.raises(ValueError):
        drv.setup(positions)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_mode1(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = pack(
        [
            torch.tensor([[0.0, 0.0, +1.0], [0.0, 0.0, -1.0]], **dd),
            torch.tensor([[0.0, 0.0, 2.0]], **dd),
        ],
        return_mask=False,
    )
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB, batch_mode=1)

    drv = BaseIntDriverPytorch(numbers, GFN1_XTB, ihelp, **dd)
    drv.setup(positions)

    assert drv._basis_batch is not None
    assert len(drv._basis_batch) == 2

    assert drv._positions_batch is not None
    assert len(drv._positions_batch) == 2

    assert drv._positions_batch[0].shape == (2, 3)
    assert (drv._positions_batch[0] == positions[0, :, :]).all()
    assert drv._positions_batch[1].shape == (1, 3)
    assert (drv._positions_batch[1] == positions[1, 0, :]).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_mode1_mask(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions, mask = pack(
        [
            torch.tensor([[0.0, 0.0, +1.0], [0.0, 0.0, -1.0]], **dd),
            torch.tensor([[0.0, 0.0, 2.0]], **dd),
        ],
        return_mask=True,
    )
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB, batch_mode=1)

    drv = BaseIntDriverPytorch(numbers, GFN1_XTB, ihelp, **dd)
    drv.setup(positions, mask=mask)

    assert drv._basis_batch is not None
    assert len(drv._basis_batch) == 2

    assert drv._positions_batch is not None
    assert len(drv._positions_batch) == 2

    assert drv._positions_batch[0].shape == (2, 3)
    assert (drv._positions_batch[0] == positions[0, :, :]).all()
    assert drv._positions_batch[1].shape == (1, 3)
    assert (drv._positions_batch[1] == positions[1, 0, :]).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_mode2(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = torch.zeros((2, 2, 3), **dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB, batch_mode=2)

    drv = BaseIntDriverPytorch(numbers, GFN1_XTB, ihelp, **dd)
    drv.setup(positions)

    assert drv._basis_batch is not None
    assert len(drv._basis_batch) == 2

    assert drv._positions_batch is not None
    assert len(drv._positions_batch) == 2
