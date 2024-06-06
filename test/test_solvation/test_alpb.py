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
Test the Analytical linearized Poisson-Boltzmann model.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.data import VDW_D3

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.components.interactions.solvation import alpb
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

opts = {
    "f_atol": 1e-10,
    "x_atol": 1e-10,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
def test_gb_single(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    dielectric_constant = torch.tensor(dielectric_constant, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charges = sample["charges"].to(**dd)
    ref = sample["energies"].to(**dd)

    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)
    cache = gb.get_cache(numbers, positions)
    energies = gb.get_atom_energy(charges, cache)

    assert pytest.approx(energies.cpu(), abs=tol) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
def test_gb_still_single(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    dc = torch.tensor(dielectric_constant, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charges = sample["charges"].to(**dd)
    rvdw = VDW_D3.to(**dd)[numbers]
    ref = sample["energies_still"].to(**dd)

    gb = alpb.GeneralizedBorn(numbers, dc, kernel="still", rvdw=rvdw, alpb=False, **dd)
    cache = gb.get_cache(numbers, positions)
    energies = gb.get_atom_energy(charges, cache)

    assert pytest.approx(energies.cpu(), abs=tol) == ref.cpu()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gb_scf(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = 1e-3 if dtype == torch.float else 1e-5
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["gsolv"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, **dd)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)

    calc_vac = Calculator(numbers, par, opts=opts, **dd)
    calc_sol = Calculator(numbers, par, interaction=[gb], opts=opts, **dd)

    results_vac = calc_vac.singlepoint(positions, charges)
    results_sol = calc_sol.singlepoint(positions, charges)

    gsolv = results_sol.scf - results_vac.scf

    assert pytest.approx(ref.cpu(), abs=tol) == gsolv.cpu()
