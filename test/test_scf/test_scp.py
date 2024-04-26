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
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb.constants import labels
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

opts = {
    "verbosity": 0,
    "maxiter": 300,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}

device = None


def single(
    dtype: torch.dtype,
    name: str,
    mixer: str,
    tol: float,
    scp_mode: str,
    scf_mode: str,
) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["escf"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "f_atol": tol,
            "x_atol": tol,
            "int_driver": "pytorch",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "LYS_xao"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single_medium(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    """Test a few larger system."""
    tol = sqrt(torch.finfo(dtype).eps) * 50
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["S2", "LYS_xao_dist"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single_difficult(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    """These systems do not reproduce tblite energies to high accuracy."""
    tol = 5e-3
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60", "vancoh2"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single_large(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


def batched(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    mixer: str,
    scp_mode: str,
    scf_mode: str,
    tol: float,
) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample[0]["numbers"].to(device),
            sample[1]["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"].to(**dd),
            sample[1]["positions"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample[0]["escf"].to(**dd),
            sample[1]["escf"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "f_atol": tol,
            "x_atol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "broyden", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full", "implicit"])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50

    # full gradient tracking (from TBMaLT) has no Broyden implementation
    if scf_mode == "full" and mixer == "broyden":
        return

    batched(dtype, name1, name2, mixer, scp_mode, scf_mode, tol)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("name3", ["SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_batch_three(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    mixer: str,
    scp_mode: str,
    scf_mode: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name1], samples[name2], samples[name3]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
            sample[2]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"],
            sample[1]["positions"],
            sample[2]["positions"],
        )
    ).type(dtype)
    ref = batch.pack(
        (
            sample[0]["escf"],
            sample[1]["escf"],
            sample[2]["escf"],
        )
    ).type(dtype)
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.1 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "f_atol": tol,
            "x_atol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)
