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
Run tests for energy contribution from halogen bond correction.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import new_halogen
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["br2nh3", "br2nh2o", "br2och2", "finch"])
def test_small(dtype: torch.dtype, name: str) -> None:
    """
    Test the halogen bond correction for small molecules taken from
    the tblite test suite.
    """
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["energy"].to(**dd)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == torch.sum(energy).cpu()


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["tmpda", "tmpda_mod"])
def test_large(dtype: torch.dtype, name: str) -> None:
    """
    TMPDA@XB-donor from S30L (15AB). Contains three iodine donors and two
    nitrogen acceptors. In the modified version, one I is replaced with
    Br and one O is added in order to obtain different donors and acceptors.
    """
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["energy"].to(**dd)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == torch.sum(energy).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_xb(dtype: torch.dtype) -> None:
    """Test system without halogen bonds."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["LYS_xao"]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["energy"].to(**dd)

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == torch.sum(energy).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_beyond_cutoff(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([7, 35], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
        ],
        **dd,
    )

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    assert pytest.approx(0.0) == torch.sum(energy).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["br2nh3", "br2och2"])
@pytest.mark.parametrize("name2", ["finch", "tmpda"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
        [
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
        ],
    )

    xb = new_halogen(numbers, par, **dd)
    if xb is None:
        assert False

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)
    energy = xb.get_energy(positions, cache)
    e = torch.sum(energy, dim=-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == e.cpu()
