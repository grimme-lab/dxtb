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
Run tests for singlepoint calculation with read from coord file.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pytest
import torch
from tad_mctc import read, read_chrg
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

slist = ["H2", "H2O", "CH4", "SiH4"]
slist_large = ["LYS_xao", "C60", "vancoh2", "AD7en+"]

opts = {
    "verbosity": 0,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}


def single(dtype: torch.dtype, name: str, gfn: str, scf_mode: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read(Path(base, "coord"), **dd)
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    ref = samples[name][f"e{gfn}"].to(**dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charge)
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_single_gfn1(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    single(dtype, name, "gfn1", scf_mode)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_single_gfn2(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    single(dtype, name, "gfn2", scf_mode)


##############################################################################


def single_large(
    dtype: torch.dtype, name: str, gfn: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read(Path(base, "coord"), **dd)
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    ref = samples[name][f"e{gfn}"].to(**dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charge)
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_single_large_gfn1(
    dtype: torch.dtype, name: str, scf_mode: str
) -> None:
    single_large(dtype, name, "gfn1", scf_mode)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_single_large_gfn2(
    dtype: torch.dtype, name: str, scf_mode: str
) -> None:
    single_large(dtype, name, "gfn2", scf_mode)


##############################################################################


def batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    gfn: str,
    scf_mode: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions, charge = [], [], []
    for name in [name1, name2, name3]:
        base = Path(Path(__file__).parent, "mols", name)
        nums, pos = read(Path(base, "coord"), **dd)
        chrg = read_chrg(Path(base, ".CHRG"), **dd)

        numbers.append(nums)
        positions.append(pos)
        charge.append(chrg)

    numbers = pack(numbers)
    positions = pack(positions)
    charge = pack(charge)
    ref = pack(
        [
            samples[name1][f"e{gfn}"].to(**dd),
            samples[name2][f"e{gfn}"].to(**dd),
            samples[name3][f"e{gfn}"].to(**dd),
        ]
    )

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charge)
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "H2O"])
@pytest.mark.parametrize("name2", ["H2", "CH4"])
@pytest.mark.parametrize("name3", ["H2", "SiH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    batch(dtype, name1, name2, name3, "gfn1", scf_mode)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "H2O"])
@pytest.mark.parametrize("name2", ["H2", "CH4"])
@pytest.mark.parametrize("name3", ["H2", "SiH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch_gfn2(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    batch(dtype, name1, name2, name3, "gfn2", scf_mode)


##############################################################################


def batch_large(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    gfn: str,
    scf_mode: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions, charge = [], [], []
    for name in [name1, name2, name3]:
        base = Path(Path(__file__).parent, "mols", name)

        nums, pos = read(Path(base, "coord"))
        chrg = read_chrg(Path(base, ".CHRG"))

        numbers.append(torch.tensor(nums, dtype=torch.long, device=DEVICE))
        positions.append(torch.tensor(pos, **dd))
        charge.append(torch.tensor(chrg, **dd))

    numbers = pack(numbers)
    positions = pack(positions)
    charge = pack(charge)
    ref = pack(
        [
            samples[name1][f"e{gfn}"].to(**dd),
            samples[name2][f"e{gfn}"].to(**dd),
            samples[name3][f"e{gfn}"].to(**dd),
        ]
    )

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charge)
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["CH4"])
@pytest.mark.parametrize("name3", ["LYS_xao"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch_large(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    batch_large(dtype, name1, name2, name3, "gfn1", scf_mode)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["CH4"])
@pytest.mark.parametrize("name3", ["LYS_xao"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch_large_gfn2(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    batch_large(dtype, name1, name2, name3, "gfn2", scf_mode)


##############################################################################


def uhf_single(dtype: torch.dtype, name: str, gfn: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read(
        Path(base, "coord"), **dd, raise_padding_warning=False
    )
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    ref = samples[name][f"e{gfn}"].to(**dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.energy(positions, charge)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == result.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "NO2"])
def test_uhf_single_gfn1(dtype: torch.dtype, name: str) -> None:
    uhf_single(dtype, name, "gfn1")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "NO2"])
def test_uhf_single_gfn2(dtype: torch.dtype, name: str) -> None:
    uhf_single(dtype, name, "gfn2")
