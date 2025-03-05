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
Test for SCF.
Reference values obtained for:
- GFN1: tblite 0.2.1, disabling repulsion and dispersion
- GFN2: tblite 0.4.0, `electronic energy`
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}

##############################################################################


def single(dtype: torch.dtype, name: str, gfn: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[f"e{gfn}"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_single_gfn1(dtype: torch.dtype, name: str) -> None:
    single(dtype, name, "gfn1")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "CH4", "SiH4"])
def test_single_gfn2(dtype: torch.dtype, name: str) -> None:
    single(dtype, name, "gfn2")


##############################################################################


def single_medium(dtype: torch.dtype, name: str, mixer: str, gfn: str) -> None:
    """Test a few larger system."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[f"e{gfn}"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "fermi_partition": "atomic",
            "maxiter": 300,
            "mixer": mixer,
            "scp_mode": "potential",
            "f_atol": tol,
            "x_atol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "name", ["PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01", "LYS_xao", "C60"]
)
@pytest.mark.parametrize("mixer", ["anderson", "broyden", "simple"])
def test_single_medium_gfn1(dtype: torch.dtype, name: str, mixer: str) -> None:
    single_medium(dtype, name, mixer, "gfn1")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "name", ["PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01", "LYS_xao", "C60"]
)
@pytest.mark.parametrize("mixer", ["anderson", "broyden", "simple"])
def test_single_medium_gfn2(dtype: torch.dtype, name: str, mixer: str) -> None:
    single_medium(dtype, name, mixer, "gfn2")


##############################################################################


def single_difficult(
    dtype: torch.dtype, name: str, gfn: str, tol: float
) -> None:
    """Test a few larger system (only float32 within tolerance)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[f"e{gfn}"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "f_atol": 1e-5 if dtype == torch.float else 1e-7,
            "x_atol": 1e-5 if dtype == torch.float else 1e-7,
            "damp": 0.5,  # simple mixing
            "maxiter": 400,  #  simple mixing
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["S2"])
def test_single_difficult_1_gfn1(dtype: torch.dtype, name: str) -> None:
    single_difficult(dtype, name, "gfn1", tol=1e-2)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["S2"])
def test_single_difficult_1_gfn2(dtype: torch.dtype, name: str) -> None:
    single_difficult(dtype, name, "gfn2", tol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LYS_xao_dist"])
def test_single_difficult_2_gfn1(dtype: torch.dtype, name: str) -> None:
    single_difficult(dtype, name, "gfn1", tol=1e-3)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LYS_xao_dist"])
def test_single_difficult_2_gfn2(dtype: torch.dtype, name: str) -> None:
    single_difficult(dtype, name, "gfn2", tol=1e-3)


##############################################################################


def single_large(dtype: torch.dtype, name: str, gfn: str) -> None:
    """Test a large systems (only float32 as they take some time)."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[f"e{gfn}"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    options = dict(
        opts,
        **{
            "damp": 0.05,  # simple mixing
            "maxiter": 300,  #  simple mixing
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large_gfn1(dtype: torch.dtype, name: str) -> None:
    single_large(dtype, name, "gfn1")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large_gfn2(dtype: torch.dtype, name: str) -> None:
    single_large(dtype, name, "gfn2")


##############################################################################


def batch(dtype: torch.dtype, name1: str, name2: str, gfn: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name1], samples[name2]
    numbers = pack(
        (
            sample[0]["numbers"].to(DEVICE),
            sample[1]["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample[0]["positions"].to(**dd),
            sample[1]["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample[0][f"e{gfn}"].to(**dd),
            sample[1][f"e{gfn}"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0], **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
def test_batch_gfn1(dtype: torch.dtype, name1: str, name2: str) -> None:
    batch(dtype, name1, name2, "gfn1")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
def test_batch_gfn2(dtype: torch.dtype, name1: str, name2: str) -> None:
    batch(dtype, name1, name2, "gfn2")


##############################################################################


def batch_three(
    dtype: torch.dtype, name1: str, name2: str, name3: str, gfn: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name1], samples[name2], samples[name3]
    numbers = pack(
        (
            sample[0]["numbers"].to(DEVICE),
            sample[1]["numbers"].to(DEVICE),
            sample[2]["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample[0]["positions"].to(**dd),
            sample[1]["positions"].to(**dd),
            sample[2]["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample[0][f"e{gfn}"].to(**dd),
            sample[1][f"e{gfn}"].to(**dd),
            sample[2][f"e{gfn}"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        assert False

    calc = Calculator(numbers, par, opts=opts, **dd)

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol) == res.cpu()


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("name3", ["SiH4"])
def test_batch_three_gfn1(
    dtype: torch.dtype, name1: str, name2: str, name3: str
) -> None:
    batch_three(dtype, name1, name2, name3, "gfn1")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("name3", ["SiH4"])
def test_batch_three_gfn2(
    dtype: torch.dtype, name1: str, name2: str, name3: str
) -> None:
    batch_three(dtype, name1, name2, name3, "gfn2")


##############################################################################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("mixer", ["anderson", "broyden", "simple"])
def test_batch_special(dtype: torch.dtype, mixer: str) -> None:
    """
    Test case for https://github.com/grimme-lab/dxtb/issues/67.

    Note that the tolerance for the energy is quite high because atoms always
    show larger deviations w.r.t. the tblite reference. Secondly, this test
    should check if the overcounting in the IndexHelper and the corresponing
    additional padding upon spreading is prevented.
    """
    tol = 1e-2  # atoms show larger deviations
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([[2, 2], [17, 0]], device=DEVICE)
    positions = pack(
        [
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd),
            torch.tensor([[0.0, 0.0, 0.0]], **dd),
        ]
    )
    chrg = torch.tensor([0.0, 0.0], **dd)
    ref = torch.tensor([-2.8629311088577, -4.1663539440167], **dd)

    options = dict(opts, **{"mixer": mixer})
    calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)

    result = calc.singlepoint(positions, chrg)
    assert pytest.approx(ref.cpu(), abs=tol) == result.scf.sum(-1).cpu()
