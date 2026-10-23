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
Test that forces do not reuse component caches built for another positions
tensor (or without a graph), even when the positions have the same values.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb.calculators import AutogradCalculator

from ...conftest import DEVICE

# GFN2's multipole integrals need libcint
methods = [(GFN1_XTB, labels.INTDRIVER_PYTORCH)]
if has_libcint is True:
    methods += [
        (GFN1_XTB, labels.INTDRIVER_LIBCINT),
        (GFN2_XTB, labels.INTDRIVER_LIBCINT),
    ]


def _setup(dd: DD, intdriver: int) -> tuple[Tensor, Tensor, dict]:
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]],
        **dd,
    )
    opts = {"verbosity": 0, "int_driver": intdriver}
    return numbers, positions, opts


def _forces(calc: AutogradCalculator, positions: Tensor) -> Tensor:
    pos = positions.clone().requires_grad_(True)
    return calc.forces(pos).detach()


@pytest.mark.parametrize("par, intdriver", methods)
def test_forces_after_energy(par, intdriver: int) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, opts = _setup(dd, intdriver)

    ref = _forces(AutogradCalculator(numbers, par, opts=opts, **dd), positions)

    calc = AutogradCalculator(numbers, par, opts=opts, **dd)
    calc.energy(positions.clone())
    assert pytest.approx(ref.cpu(), abs=1e-10) == _forces(calc, positions).cpu()


@pytest.mark.parametrize("par, intdriver", methods)
def test_forces_twice_new_tensor(par, intdriver: int) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, opts = _setup(dd, intdriver)

    calc = AutogradCalculator(numbers, par, opts=opts, **dd)
    first = _forces(calc, positions)
    second = _forces(calc, positions)
    assert pytest.approx(first.cpu(), abs=1e-10) == second.cpu()
