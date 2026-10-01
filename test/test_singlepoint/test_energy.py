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

from dxtb import Calculator
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples

slist = ["H2", "H2O", "CH4", "SiH4"]
slist_large = ["LYS_xao", "C60", "vancoh2", "AD7en+"]

# Batching only needs representative triples (differing sizes, and identical
# molecules for a batch without padding); every sample is checked individually
# in the `test_single_*` tests.
batch_triples = [
    ("H2O", "SiH4", "H2"),
    ("H2", "H2", "LiH"),
    ("H2", "H2", "H2"),
]

opts = {
    "verbosity": 0,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}


def single(
    dtype: torch.dtype,
    name: str,
    gfn: str,
    scf_mode: str = labels.SCF_MODE_IMPLICIT_NON_PURE,
    int_driver: str | None = None,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read(Path(base, "coord"), **dd)
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    ref = samples[name][f"e{gfn}"].to(**dd)

    par = get_param_module(gfn, **dd)

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    if int_driver is not None:
        options["int_driver"] = int_driver
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


@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_single_gfn2_pytorch(name: str) -> None:
    """GFN2 with the multipole integrals of the PyTorch driver (no libcint)."""
    single(torch.double, name, "gfn2", int_driver="analytical")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_single_gfn0(dtype: torch.dtype, name: str) -> None:
    single(dtype, name, "gfn0")


##############################################################################


def single_large(
    dtype: torch.dtype,
    name: str,
    gfn: str,
    scf_mode: str = labels.SCF_MODE_IMPLICIT_NON_PURE,
    int_driver: str | None = None,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read(Path(base, "coord"), **dd)
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    ref = samples[name][f"e{gfn}"].to(**dd)

    par = get_param_module(gfn, **dd)

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    if int_driver is not None:
        options["int_driver"] = int_driver
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


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large_gfn0(dtype: torch.dtype, name: str) -> None:
    single_large(dtype, name, "gfn0")


##############################################################################


def batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    gfn: str,
    scf_mode: str = labels.SCF_MODE_IMPLICIT_NON_PURE,
    int_driver: str | None = None,
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

    par = get_param_module(gfn, **dd)

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    if int_driver is not None:
        options["int_driver"] = int_driver
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charge)
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1, name2, name3", batch_triples)
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch_gfn1(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    batch(dtype, name1, name2, name3, "gfn1", scf_mode)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1, name2, name3", batch_triples)
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_batch_gfn2(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    batch(dtype, name1, name2, name3, "gfn2", scf_mode)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1, name2, name3", batch_triples)
def test_batch_gfn0(
    dtype: torch.dtype, name1: str, name2: str, name3: str
) -> None:
    batch(dtype, name1, name2, name3, "gfn0")


##############################################################################


def batch_large(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    gfn: str,
    scf_mode: str = labels.SCF_MODE_IMPLICIT_NON_PURE,
    int_driver: str | None = None,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions, charge = [], [], []
    for name in [name1, name2, name3]:
        base = Path(Path(__file__).parent, "mols", name)

        nums, pos = read(Path(base, "coord"), dtype_int=torch.long, **dd)
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

    par = get_param_module(gfn, **dd)

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    if int_driver is not None:
        options["int_driver"] = int_driver
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


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["CH4"])
@pytest.mark.parametrize("name3", ["LYS_xao"])
def test_batch_large_gfn0(
    dtype: torch.dtype, name1: str, name2: str, name3: str
) -> None:
    batch_large(dtype, name1, name2, name3, "gfn0")


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

    par = get_param_module(gfn, **dd)

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


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "NO2"])
def test_uhf_single_gfn0(dtype: torch.dtype, name: str) -> None:
    uhf_single(dtype, name, "gfn0")
