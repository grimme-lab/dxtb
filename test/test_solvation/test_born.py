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
Run tests for calculation of Born radii according to the Onufriev-Bashford-Case
model. Reference values are obtained from the tblite version.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.data.radii import VDW_D3

from dxtb._src.components.interactions.solvation import born
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_psi(name: str, dtype: torch.dtype):
    """Test psi for mb16_43_01 and mb16_43_02."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["psi"].to(**dd)
    rvdw = VDW_D3.to(**dd)[numbers]

    psi = born.compute_psi(numbers, positions, rvdw)
    assert pytest.approx(ref.cpu(), rel=1e-6, abs=tol) == psi.cpu()


def test_fail_shape():
    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=DEVICE,
    )

    # rcov wrong shape
    with pytest.raises(ValueError):
        rvdw = torch.tensor([1.0], device=DEVICE)
        born.get_born_radii(numbers, positions, rvdw)

    # wrong numbers
    with pytest.raises(ValueError):
        born.get_born_radii(torch.tensor([1], device=DEVICE), positions)

    # wrong
    with pytest.raises(ValueError):
        descreening = torch.tensor([1], device=DEVICE)
        born.get_born_radii(numbers, positions, descreening=descreening)


@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_radii(name: str, dtype: torch.dtype):
    """Test Born radii for mb16_43_01 and mb16_43_02."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["born"].to(**dd)

    rads = born.get_born_radii(numbers, positions)
    assert pytest.approx(ref.cpu(), rel=1e-6, abs=tol) == rads.cpu()


@pytest.mark.parametrize("name1", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_psi_batch(name1: str, name2: str, dtype: torch.dtype):
    """Test psi for batch."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

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
    rvdw = VDW_D3.to(**dd)[numbers]

    ref = pack(
        (
            sample1["psi"].to(**dd),
            sample2["psi"].to(**dd),
        ),
    )

    psi = born.compute_psi(numbers, positions, rvdw)
    assert pytest.approx(ref.cpu(), rel=1e-6, abs=tol) == psi.cpu()


@pytest.mark.parametrize("name1", ["MB16_43_01", "MB16_43_02"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_radii_batch(name1: str, name2: str, dtype: torch.dtype):
    """Test Born radii for batch."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

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
    descreening = pack(
        (
            torch.full(sample1["numbers"].shape, 0.8, **dd),
            torch.full(sample2["numbers"].shape, 0.8, **dd),
        )
    )

    ref = pack(
        (
            sample1["born"].to(**dd),
            sample2["born"].to(**dd),
        ),
    )

    rads = born.get_born_radii(numbers, positions, descreening=descreening)
    assert pytest.approx(ref.cpu(), rel=1e-6, abs=tol) == rads.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("name", ["MB16_43_01", "SiH4"])
def test_psi_grad(name: str):
    """Test autograd of psi w.r.t to positions."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    rvdw = VDW_D3.to(**dd)[numbers]

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor):
        return born.compute_psi(numbers, p, rvdw)

    assert dgradcheck(func, pos)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["MB16_43_01", "SiH4"])
def test_radii_grad(name: str):
    """Test autograd of born radii w.r.t to positions."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor):
        return born.get_born_radii(numbers, p)

    assert dgradcheck(func, pos)
