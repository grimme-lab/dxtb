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
Run general tests for the spin-polarization interaction.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB, Calculator
from dxtb._src.components.interactions import ChargeChannel, InteractionList
from dxtb._src.components.interactions.spin import (
    LABEL_SPIN_POLARIZATION,
    SpinPolarization,
    new_spin_polarization,
)
from dxtb._src.typing.exceptions import DeviceError

numbers = torch.tensor([8, 1])


def test_factory() -> None:
    cls = new_spin_polarization(numbers, wscale=0.5, dtype=torch.float64)

    assert isinstance(cls, SpinPolarization)
    assert cls.label == LABEL_SPIN_POLARIZATION
    assert cls.dtype == torch.float64
    assert cls.wscale.dtype == torch.float64
    assert pytest.approx(0.5) == cls.wscale.item()

    # acts on magnetization channel and requires two spin channels
    assert cls.charge_channel == ChargeChannel.MAGNETIZATION
    assert cls.requires_uhf is True
    assert InteractionList(cls).requires_uhf is True
    assert InteractionList().requires_uhf is False


@pytest.mark.parametrize("uhf_mode", [False, True])
@pytest.mark.parametrize("spinpol", [False, True])
def test_nspin(uhf_mode: bool, spinpol: bool) -> None:
    interaction = [new_spin_polarization(numbers)] if spinpol else None
    calc = Calculator(
        numbers, GFN1_XTB, interaction=interaction, opts={"uhf_mode": uhf_mode}
    )

    # spin polarization automatically enables UHF
    assert calc.nspin == (2 if uhf_mode or spinpol else 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cls = new_spin_polarization(numbers)

    cls = cls.type(dtype)
    assert cls.dtype == dtype
    assert cls.spin_constants.dtype == dtype
    assert cls.wscale.dtype == dtype


def test_change_type_fail() -> None:
    cls = new_spin_polarization(numbers)

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        cls.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)
    cls = new_spin_polarization(numbers)

    cls = cls.to(device)
    assert cls.device == device


def test_change_device_fail() -> None:
    cls = new_spin_polarization(numbers)

    # trying to use setter
    with pytest.raises(AttributeError):
        cls.device = "cpu"


def test_fail_element() -> None:
    with pytest.raises(ValueError):
        new_spin_polarization(torch.tensor([87, 1]))


@pytest.mark.cuda
def test_fail_device() -> None:
    with pytest.raises(DeviceError):
        new_spin_polarization(numbers, device=torch.device("cuda"))
