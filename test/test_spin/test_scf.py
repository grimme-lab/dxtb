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
Test the SCF variants (SCF modes, convergence targets, mixers, culling) for
spin-polarized calculations.

Reference values obtained with tblite 0.7.0 (see `samples.py`).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.components.interactions.spin import new_spin_polarization
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("scf_mode", ["implicit", "full"])
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_modes(scf_mode: str, scp_mode: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = 1e-5

    sample = samples["OH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    spin = sample["spin"].to(**dd)
    ref = sample["egfn1"].to(**dd)

    opts = {"verbosity": 0, "scf_mode": scf_mode, "scp_mode": scp_mode}
    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(numbers, GFN1_XTB, interaction=[spinpol], opts=opts, **dd)

    result = calc.singlepoint(positions, spin=spin)
    assert pytest.approx(ref.cpu(), abs=tol) == result.total.sum(-1).cpu()


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_culling(scp_mode: str, mixer: str) -> None:
    """
    Batched full SCF with culling of converged systems. The (larger) closed
    shell system typically converges first.
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = 1e-5

    sample1, sample2 = samples["SiH4"], samples["OH"]
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
    spin = torch.stack([sample1["spin"], sample2["spin"]]).to(**dd)
    ref = torch.stack([sample1["egfn2"], sample2["egfn2"]]).to(**dd)

    opts = {
        "verbosity": 0,
        "maxiter": 300,
        "scf_mode": "full",
        "scp_mode": scp_mode,
        "mixer": mixer,
        "damp": 0.1 if mixer == "simple" else 0.4,
    }
    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(numbers, GFN2_XTB, interaction=[spinpol], opts=opts, **dd)

    chrg = torch.zeros(2, **dd)
    result = calc.singlepoint(positions, chrg, spin)
    assert pytest.approx(ref.cpu(), abs=tol) == result.total.sum(-1).cpu()

    # total charge and number of unpaired electrons
    q = result.charges.mono.sum(-1)
    assert pytest.approx(torch.zeros(2), abs=tol) == q[:, 0].cpu()
    assert pytest.approx(-spin.cpu(), abs=tol) == q[:, 1].cpu()


def test_fermi_closed_shell() -> None:
    """Closed-shell UHF at high electronic temperature equals restricted."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = 1e-8

    numbers = mols["H2"]["numbers"].to(DEVICE)
    positions = mols["H2"]["positions"].to(**dd)

    opts = {"verbosity": 0, "fermi_etemp": 10000.0}
    restricted = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    result_rhf = restricted.singlepoint(positions)

    opts["uhf_mode"] = True
    unrestricted = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    result_uhf = unrestricted.singlepoint(positions)

    assert (
        pytest.approx(result_rhf.total.cpu(), abs=tol) == result_uhf.total.cpu()
    )
    assert (
        pytest.approx(result_rhf.occupation.cpu(), abs=tol)
        == result_uhf.occupation.cpu()
    )
