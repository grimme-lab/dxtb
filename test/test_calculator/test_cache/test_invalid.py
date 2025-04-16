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
# pylint: disable=protected-access
from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB
from dxtb._src.typing import DD, Tensor
from dxtb.calculators import AutogradCalculator

from ...conftest import DEVICE

opts = {"cache_enabled": True, "verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_invalid_newpos(dtype: torch.dtype) -> None:
    """Test that the cache is invalidated when new positions are used."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    positions1.requires_grad_(True)

    positions2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], **dd)
    positions2.requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop1 = calc.get_forces(positions1)
    assert calc._ncalcs == 1
    assert isinstance(prop1, Tensor)

    # cache active for first calc
    prop1 = calc.get_forces(positions1)
    assert calc._ncalcs == 1
    assert isinstance(prop1, Tensor)

    # now run different positions
    prop2 = calc.get_forces(positions2)
    assert calc._ncalcs == 2
    assert isinstance(prop2, Tensor)

    assert not torch.equal(prop1, prop2)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_invalid_requiresgrad(dtype: torch.dtype) -> None:
    """Test that the cache is invalidated when requires_grad is changed."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    positions1.requires_grad_(False)  # no grad!

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop1 = calc.get_energy(positions1)
    assert calc._ncalcs == 1
    assert isinstance(prop1, Tensor)

    # cache active for first calc
    prop1 = calc.get_energy(positions1)
    assert calc._ncalcs == 1
    assert isinstance(prop1, Tensor)

    # now run with different positions (gradient)
    positions1.requires_grad_(True)
    prop2 = calc.get_energy(positions1)
    assert calc._ncalcs == 2
    assert isinstance(prop2, Tensor)

    assert torch.equal(prop1, prop2)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_invalid_inplace(dtype: torch.dtype) -> None:
    """Test that the cache is invalidated when inplace changes are made."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop1 = calc.get_energy(positions1)
    assert calc._ncalcs == 1
    assert isinstance(prop1, Tensor)

    # cache active for first calc
    prop1 = calc.get_energy(positions1)
    assert calc._ncalcs == 1
    assert isinstance(prop1, Tensor)

    # now run with different positions (inplace change, increases `_version`)
    positions1[0, 0] = 1.0
    prop2 = calc.get_energy(positions1)
    assert calc._ncalcs == 2
    assert isinstance(prop2, Tensor)

    assert not torch.equal(prop1, prop2)
