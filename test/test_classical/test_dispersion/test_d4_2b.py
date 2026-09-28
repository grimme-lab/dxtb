# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Testing dispersion energy for the 2-body Dispersion in GFN0-xTB.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb._src.typing import DD
from dxtb.calculators import GFN0Calculator

from ...conftest import DEVICE
from .samples import samples

sample_list = [
    "LiH",
    "SiH4",
    "PbH4-BiH3",
    "C6H5I-CH3SH",
    "MB16_43_01",
]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    single(name, dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large(name: str, dtype: torch.dtype) -> None:
    single(name, dtype)


def single(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = sample["edisp_2b"].to(**dd)

    calc = get_calculator(numbers, dtype)
    result = calc.singlepoint(positions, chrg=charge)
    energy = result.cenergies["DispersionD4"].sum(-1)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    batch(name1, name2, dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["vancoh2"])
def test_batch_large(name1: str, name2: str, dtype: torch.dtype) -> None:
    batch(name1, name2, dtype)


def batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    charge = positions.new_zeros(numbers.shape[0])
    ref = torch.stack(
        [
            sample1["edisp_2b"].to(**dd),
            sample2["edisp_2b"].to(**dd),
        ]
    )

    calc = get_calculator(numbers, dtype)
    result = calc.singlepoint(positions, chrg=charge)
    atomwise = result.cenergies["DispersionD4"]
    energy = atomwise.sum(-1)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()
    assert torch.count_nonzero(atomwise[0, sample1["numbers"].numel() :]) == 0


def get_calculator(numbers: torch.Tensor, dtype: torch.dtype) -> GFN0Calculator:
    return GFN0Calculator(
        numbers,
        opts={
            "verbosity": 0,
            "exclude": ["ies", "rep", "srb", "scf"],
        },
        device=DEVICE,
        dtype=dtype,
    )
