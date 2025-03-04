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

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..utils import load_from_npz
from .samples_charged import samples

opts = {
    "verbosity": 0,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}

ref_grad = np.load("test/test_scf/grad.npz")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "name", ["Ag2Cl22-", "Al3+Ar6", "AD7en+", "C2H4F+", "ZnOOH-"]
)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_single(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[f"e{gfn}"]
    chrg = sample["charge"].to(**dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    calc = Calculator(numbers, par, opts=opts, **dd)
    results = calc.singlepoint(positions, chrg)
    res = results.scf.sum(-1)

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["Ag2Cl22-", "Al3+Ar6", "C2H4F+", "ZnOOH-"])
def test_grad(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd).detach()
    pos = positions.clone().requires_grad_(True)
    chrg = sample["charge"].to(**dd)

    # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
    ref = load_from_npz(ref_grad, name, dtype)

    options = dict(
        opts,
        **{
            "exclude": ["rep", "disp", "hal"],
            "maxiter": 50,
            "f_atol": 1.0e-5,
            "x_atol": 1.0e-5,
        },
    )
    calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)
    result = calc.singlepoint(pos, chrg)
    energy = result.scf.sum(-1)

    (gradient,) = torch.autograd.grad(energy, pos)
    assert pytest.approx(gradient.cpu(), abs=tol, rel=1e-5) == ref.cpu()
