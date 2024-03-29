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
Test for SCF with charged samples.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb._types import DD
from dxtb.param import GFN1_XTB as par
from dxtb.xtb.calculator import Calculator

from ..utils import load_from_npz
from .samples_charged import samples

opts = {"verbosity": 0}

ref_grad = np.load("test/test_scf/grad.npz")

device = None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["Ag2Cl22-", "Al3+Ar6", "AD7en+", "C2H4F+", "ZnOOH-"])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["escf"].item()
    chrg = sample["charge"].to(**dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    results = calc.singlepoint(numbers, positions, chrg)

    assert pytest.approx(ref, abs=tol, rel=tol) == results.scf.sum(-1).item()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["Ag2Cl22-", "Al3+Ar6", "C2H4F+", "ZnOOH-"])
def test_grad(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd).detach()
    positions.requires_grad_(True)
    chrg = sample["charge"].to(**dd)

    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(
        opts,
        **{
            "exclude": ["rep", "disp", "hal"],
            "maxiter": 50,
            "f_atol": 1.0e-6,
            "x_atol": 1.0e-6,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)
    result = calc.singlepoint(numbers, positions, chrg)
    energy = result.scf.sum(-1)

    gradient = torch.autograd.grad(
        energy,
        positions,
    )[0]

    assert pytest.approx(gradient, abs=tol, rel=1e-5) == ref
