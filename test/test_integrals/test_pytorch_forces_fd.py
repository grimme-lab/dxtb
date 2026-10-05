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
Autograd forces of GFN1-xTB and GFN2-xTB with the PyTorch integral driver against central
finite differences of the energy, for several algorithms.
"""

from __future__ import annotations

import pytest
import torch

import dxtb
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("algorithm", ["md", "os"])
@pytest.mark.parametrize("name", ["H2O"])
@pytest.mark.parametrize(
    "cls", [dxtb.calculators.GFN1Calculator, dxtb.calculators.GFN2Calculator]
)
def test_forces_match_finite_differences(
    cls, name: str, algorithm: str
) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_algorithm": algorithm,
        "scf_mode": "full",
        "f_atol": 1e-11,
        "x_atol": 1e-11,
    }

    def energy(pos: torch.Tensor) -> torch.Tensor:
        return cls(numbers, opts=opts, **dd).get_energy(pos)

    pos = positions.detach().clone().requires_grad_(True)
    forces = cls(numbers, opts=opts, **dd).get_forces(pos)

    step = 1e-4
    numeric = torch.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.detach().clone()
            minus = positions.detach().clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numeric[atom, axis] = -(energy(plus) - energy(minus)) / (2 * step)

    assert torch.allclose(forces, numeric, atol=1e-7, rtol=0.0)
