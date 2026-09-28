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
Run tests for the short range bond correction energy contribution.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN0_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import new_srb
from dxtb._src.typing import DD

from ...conftest import DEVICE
from .samples import samples

sample_list = [
    "LiH",
    "SiH4",
    "PbH4-BiH3",
    "MB16_43_01",
    "Ag2Cl22-",
    "Rn",
    "LYS_xao",
    "NO2",
]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test neutral, charged, molecular, and atomic systems."""
    single(dtype, name)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large(dtype: torch.dtype, name: str) -> None:
    """Test the SRB correction energy for a large system."""
    single(dtype, name)


def single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["esrb"].to(**dd)

    srb = new_srb(numbers, par, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = srb.get_cache(numbers, ihelp)
    energy = srb.get_energy(positions, cache).sum(-1)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH", "SiH4"])
@pytest.mark.parametrize("name2", ["MB16_43_01", "Ag2Cl22-"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Test heterogeneous padded batches, including a charged system."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

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
    ref = torch.stack(
        (
            sample1["esrb"].to(**dd),
            sample2["esrb"].to(**dd),
        )
    )

    srb = new_srb(numbers, par, **dd)
    assert srb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = srb.get_cache(numbers, ihelp)
    atomwise = srb.get_energy(positions, cache)
    energy = atomwise.sum(-1)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == energy.cpu()

    nat1 = sample1["numbers"].numel()
    assert torch.count_nonzero(atomwise[0, nat1:]) == 0
