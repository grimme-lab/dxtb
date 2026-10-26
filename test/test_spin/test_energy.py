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
Test the SCF energy and charges of unrestricted (UHF) and spin-polarized
calculations.

Reference values obtained with tblite 0.7.0 (see `samples.py`).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.components.interactions.spin import new_spin_polarization
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

opts = {"verbosity": 0}

sample_list_gfn1 = ["H2+", "OH"]
sample_list_gfn2 = ["LiH", "OH", "SiH4", "C2H5O"]


def single(
    dtype: torch.dtype, name: str, gfn: str, mode: str, qtol: float = 1e-3
) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1e-5 if dtype == torch.double else 1e-4

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    chrg = sample["charge"].to(**dd)
    spin = sample["spin"].to(**dd)
    ref = sample[f"e{gfn}{mode}"].to(**dd)
    ref_q = sample[f"q{gfn}{mode}"].to(**dd)

    par = GFN1_XTB if gfn == "gfn1" else GFN2_XTB

    # UHF is either requested explicitly or by the spin polarization
    if mode == "_uhf":
        calc = Calculator(numbers, par, opts={**opts, "uhf_mode": True}, **dd)
    else:
        spinpol = new_spin_polarization(numbers, **dd)
        calc = Calculator(numbers, par, interaction=[spinpol], opts=opts, **dd)

    result = calc.singlepoint(positions, chrg, spin)
    assert result.nspin == 2

    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()

    # atomic charges and magnetization
    q = torch.stack(
        [
            calc.ihelp.reduce_orbital_to_atom(qorb)
            for qorb in result.charges.mono
        ]
    )
    assert pytest.approx(ref_q.cpu(), abs=qtol) == q.cpu()

    # sum rules: total charge and number of unpaired electrons
    assert pytest.approx(chrg.cpu(), abs=tol) == q[0].sum(-1).cpu()
    assert pytest.approx(-spin.cpu(), abs=tol) == q[1].sum(-1).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list_gfn1)
@pytest.mark.parametrize("mode", ["", "_uhf"])
def test_single_gfn1(dtype: torch.dtype, name: str, mode: str) -> None:
    single(
        dtype, name, "gfn1", mode, qtol=1e-4 if dtype == torch.double else 1e-3
    )


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["CrCp2"])
@pytest.mark.parametrize("mode", ["", "_uhf"])
def test_single_gfn1_large(dtype: torch.dtype, name: str, mode: str) -> None:
    single(dtype, name, "gfn1", mode)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list_gfn2)
@pytest.mark.parametrize("mode", ["", "_uhf"])
def test_single_gfn2(dtype: torch.dtype, name: str, mode: str) -> None:
    single(
        dtype, name, "gfn2", mode, qtol=1e-4 if dtype == torch.double else 1e-3
    )


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list_gfn1)
@pytest.mark.parametrize("name2", sample_list_gfn1)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1e-5 if dtype == torch.double else 1e-4

    sample1, sample2 = samples[name1], samples[name2]
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
    chrg = torch.stack([sample1["charge"], sample2["charge"]]).to(**dd)
    spin = torch.stack([sample1["spin"], sample2["spin"]]).to(**dd)
    ref = torch.stack([sample1["egfn1"], sample2["egfn1"]]).to(**dd)

    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(numbers, GFN1_XTB, interaction=[spinpol], opts=opts, **dd)
    result = calc.singlepoint(positions, chrg, spin)

    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_closed_shell(dtype: torch.dtype) -> None:
    """Closed-shell UHF calculations reproduce the restricted result."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1e-5 if dtype == torch.double else 1e-4

    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    calc = Calculator(numbers, GFN2_XTB, opts=opts, **dd)
    restricted = calc.singlepoint(positions)
    assert restricted.nspin == 1

    calc = Calculator(numbers, GFN2_XTB, opts={**opts, "uhf_mode": True}, **dd)
    uhf = calc.singlepoint(positions)

    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(numbers, GFN2_XTB, interaction=[spinpol], opts=opts, **dd)
    spin_polarized = calc.singlepoint(positions)

    for result in (uhf, spin_polarized):
        assert result.nspin == 2
        assert (
            pytest.approx(restricted.total.cpu(), abs=tol) == result.total.cpu()
        )

        # alpha and beta channel are identical
        assert (
            pytest.approx(result.density[0].cpu(), abs=tol)
            == result.density[1].cpu()
        )
        assert (
            pytest.approx(restricted.density.cpu(), abs=tol)
            == result.density.sum(-3).cpu()
        )
