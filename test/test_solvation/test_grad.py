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
Test gradient of the Analytical linearized Poisson-Boltzmann model.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.components.interactions.solvation import alpb
from dxtb.constants import labels
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD
from dxtb.xtb import Calculator

from .samples import samples

opts = {
    "f_atol": 1e-10,
    "x_atol": 1e-10,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

device = None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gb_scf_grad(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = 1e-3 if dtype == torch.float else 1e-5
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    positions.requires_grad_(True)
    ref = sample["gradient"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, **dd)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)

    calc = Calculator(numbers, par, interaction=[gb], opts=opts, **dd)

    results = calc.singlepoint(positions, charges)
    energy = results.scf.sum(-1)

    # autograd
    energy.backward()
    assert positions.grad is not None
    autograd = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref, abs=tol) == autograd
