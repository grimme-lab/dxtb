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
Test for self-consistent D4 dispersion energy
Reference values obtained with tblite 0.4.0 disabling all other interactions.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN2_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ...conftest import DEVICE
from .samples import samples

slist = ["LiH", "SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "exclude": ["es2", "es3"],
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["edisp_d4sc"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    calc = Calculator(numbers, GFN2_XTB, opts=opts, **dd)

    result = calc.singlepoint(positions, charges)
    d4sc = calc.interactions.get_interaction("DispersionD4SC")
    cache = d4sc.get_cache(numbers=numbers, positions=positions)

    edisp = d4sc.get_energy(result.charges, cache, calc.ihelp)
    assert pytest.approx(ref.cpu(), abs=10 * tol, rel=tol) == edisp.cpu()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", slist)
@pytest.mark.parametrize("name2", ["LiH", "MB16_43_01"])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = (samples[name1], samples[name2])
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["edisp_d4sc"].to(**dd),
            sample2["edisp_d4sc"].to(**dd),
        )
    )

    calc = Calculator(numbers, GFN2_XTB, opts=opts, **dd)

    result = calc.singlepoint(positions)
    d4sc = calc.interactions.get_interaction("DispersionD4SC")
    cache = d4sc.get_cache(numbers=numbers, positions=positions)

    edisp = d4sc.get_energy(result.charges, cache, calc.ihelp)
    assert pytest.approx(ref.cpu(), abs=10 * tol, rel=tol) == edisp.cpu()
