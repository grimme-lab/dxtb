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
finite differences. For GFN2, the gradient also flows through the multipole
integrals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import Calculator
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import DD, Literal, Tensor

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples


def energy_of_scaled_exponents(
    gfn: Literal["gfn1", "gfn2"],
    numbers: Tensor,
    positions: Tensor,
    scale: Tensor,
    driver: str,
    dd: DD,
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
        par = get_param_module(gfn, **dd)
        calc = Calculator(numbers, par, opts=opts, **dd)
        return calc.get_energy(positions)
    finally:
        Basis.create_cgtos = original  # type: ignore[method-assign]


@pytest.mark.parametrize(
    "gfn, name", [("gfn1", "LiH"), ("gfn1", "H2O"), ("gfn2", "LiH")]
)
def test_exponent_gradient_matches_fd(
    gfn: Literal["gfn1", "gfn2"], name: str
) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    scale = torch.tensor(1.0, **dd, requires_grad=True)
    energy = energy_of_scaled_exponents(
        gfn, numbers, positions, scale, "pytorch", dd
    )
    (grad,) = torch.autograd.grad(energy, scale)

    step = 1e-4
    plus = energy_of_scaled_exponents(
        gfn, numbers, positions, torch.tensor(1.0 + step, **dd), "pytorch", dd
    )
    minus = energy_of_scaled_exponents(
        gfn, numbers, positions, torch.tensor(1.0 - step, **dd), "pytorch", dd
    )
    grad_fd = (plus - minus) / (2 * step)

    assert torch.allclose(grad, grad_fd, atol=1e-6, rtol=1e-5)
