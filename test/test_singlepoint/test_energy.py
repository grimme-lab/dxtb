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

from dxtb._types import DD
from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0}

device = None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4", "SiH4", "LYS_xao"])
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    charge = torch.tensor(charge).type(dtype)

    ref = samples[name]["etot"].item()

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, charge)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1).item()


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60", "vancoh2", "AD7en+"])
def test_single_large(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    charge = torch.tensor(charge).type(dtype)

    ref = samples[name]["etot"].item()

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, charge)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1).item()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "H2O"])
@pytest.mark.parametrize("name2", ["H2", "CH4"])
@pytest.mark.parametrize("name3", ["H2", "SiH4", "LYS_xao"])
def test_batch(dtype: torch.dtype, name1: str, name2: str, name3: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    numbers, positions, charge = [], [], []
    for name in [name1, name2, name3]:
        base = Path(Path(__file__).parent, "mols", name)

        nums, pos = read_coord(Path(base, "coord"))
        chrg = read_chrg(Path(base, ".CHRG"))

        numbers.append(torch.tensor(nums, dtype=torch.long))
        positions.append(torch.tensor(pos).type(dtype))
        charge.append(torch.tensor(chrg).type(dtype))

    numbers = batch.pack(numbers)
    positions = batch.pack(positions)
    charge = batch.pack(charge)
    ref = batch.pack(
        [
            samples[name1]["etot"].to(**dd),
            samples[name2]["etot"].to(**dd),
            samples[name3]["etot"].to(**dd),
        ]
    )

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, charge)
    assert torch.allclose(ref, result.total.sum(-1), atol=tol, rtol=tol)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "NO2"])
def test_uhf_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    base = Path(Path(__file__).parent, "mols", name)

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    charge = torch.tensor(charge).type(dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)

    ref = samples[name]["etot"].item()
    result = calc.singlepoint(numbers, positions, charge)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1).item()
