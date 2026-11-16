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
Test conversion between alpha/beta and charge/magnetization representation.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck

from dxtb._src.typing import DD
from dxtb._src.wavefunction import spin

from ..conftest import DEVICE, NONDET_TOL


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_conversion(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    updown = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], **dd)
    ref = torch.tensor([[1.5, 1.0, 7.0], [0.5, 3.0, -1.0]], **dd)

    magnet = spin.updown_to_charge_magnetization(updown, dim=0)
    assert pytest.approx(ref.cpu()) == magnet.cpu()

    restored = spin.charge_magnetization_to_updown(magnet, dim=0)
    assert pytest.approx(updown.cpu()) == restored.cpu()


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_dim(dim: int) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    # all dimensions have size 2, only `dim` is the spin dimension
    updown = torch.randn((2, 2, 2), **dd)
    magnet = spin.updown_to_charge_magnetization(updown, dim=dim)

    alpha, beta = updown.unbind(dim)
    assert pytest.approx((alpha + beta).cpu()) == magnet.select(dim, 0).cpu()
    assert pytest.approx((alpha - beta).cpu()) == magnet.select(dim, 1).cpu()

    restored = spin.charge_magnetization_to_updown(magnet, dim=dim)
    assert pytest.approx(updown.cpu()) == restored.cpu()


@pytest.mark.grad
def test_grad() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    # variable to be differentiated
    x = torch.randn((3, 2, 4), **dd, requires_grad=True)

    def func(x: torch.Tensor) -> torch.Tensor:
        magnet = spin.updown_to_charge_magnetization(x, dim=1)
        return spin.charge_magnetization_to_updown(magnet**2, dim=1)

    assert dgradcheck(func, x, nondet_tol=NONDET_TOL)
