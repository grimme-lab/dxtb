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
Gradient of the total energy with respect to the basis exponents against
finite differences.

Regression test: the custom backward of the analytical driver keeps the basis
as a plain Python object, so it silently dropped all gradients with respect to
the exponents and contraction coefficients (LiH: 1.7e-3 instead of 0.19).
"""

from __future__ import annotations

import pytest
import torch

import dxtb
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples


def energy_of_scaled_exponents(
    numbers: Tensor, positions: Tensor, scale: Tensor, driver: str, dd: DD
) -> Tensor:
    """Single point with every primitive exponent multiplied by ``scale``,
    standing in for a fitted basis parameter."""
    original = Basis.create_cgtos

    def patched(self):
        alphas, coeffs = original(self)
        return [a * scale for a in alphas], coeffs

    Basis.create_cgtos = patched  # type: ignore[method-assign]
    try:
        opts = {
            "verbosity": 0,
            "int_driver": driver,
            "scf_mode": "full",
            "f_atol": 1e-11,
            "x_atol": 1e-11,
        }
        calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)
        return calc.get_energy(positions)
    finally:
        Basis.create_cgtos = original  # type: ignore[method-assign]


@pytest.mark.parametrize("driver", ["autograd", "analytical"])
@pytest.mark.parametrize("name", ["LiH", "H2O"])
def test_exponent_gradient_matches_fd(name: str, driver: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    scale = torch.tensor(1.0, **dd, requires_grad=True)
    energy = energy_of_scaled_exponents(numbers, positions, scale, driver, dd)
    (grad,) = torch.autograd.grad(energy, scale)

    step = 1e-4
    plus = energy_of_scaled_exponents(
        numbers, positions, torch.tensor(1.0 + step, **dd), driver, dd
    )
    minus = energy_of_scaled_exponents(
        numbers, positions, torch.tensor(1.0 - step, **dd), driver, dd
    )
    grad_fd = (plus - minus) / (2 * step)

    assert torch.allclose(grad, grad_fd, atol=1e-6, rtol=1e-5)
