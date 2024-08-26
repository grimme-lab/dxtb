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
Run tests for building the Hamiltonian matrix.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb._src.integral.driver.pytorch import OverlapPytorch as Overlap
from dxtb._src.param import Param
from dxtb._src.typing import DD, Tensor
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian

from ..conftest import DEVICE
from ..utils import load_from_npz
from .samples import samples

small = ["C", "Rn", "H2", "LiH", "HLi", "S2", "SiH4"]
large = ["PbH4-BiH3", "LYS_xao"]

ref_h0 = np.load("test/test_hamiltonian/h0.npz")


def run(numbers: Tensor, positions: Tensor, par: Param, ref: Tensor, dd: DD) -> None:
    tol = sqrt(torch.finfo(dd["dtype"]).eps) * 10

    ihelp = IndexHelper.from_numbers(numbers, par)
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(**dd)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)

    driver.setup(positions)
    s = overlap.build(driver)

    h = h0.build(positions, s)
    assert pytest.approx(h.cpu(), abs=tol) == h.mT.cpu()
    assert pytest.approx(h.cpu(), abs=tol) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", small)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_h0, name, dtype)

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
@pytest.mark.parametrize("name2", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
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
    ref = pack(
        (
            load_from_npz(ref_h0, name1, dtype),
            load_from_npz(ref_h0, name2, dtype),
        ),
    )

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", large)
def test_large(dtype: torch.dtype, name: str) -> None:
    """Compare against reference calculated with tblite."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_h0, name, dtype)

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", large)
@pytest.mark.parametrize("name2", large)
def test_large_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
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
    ref = pack(
        (
            load_from_npz(ref_h0, name1, dtype),
            load_from_npz(ref_h0, name2, dtype),
        )
    )

    run(numbers, positions, GFN1_XTB, ref, dd)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_cn(dtype: torch.dtype) -> None:
    """Test without CN."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["H2"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = torch.tensor(
        [
            [
                -0.40142945681830,
                -0.00000000000000,
                -0.47765679842079,
                -0.03687145777483,
            ],
            [
                -0.00000000000000,
                -0.07981592633195,
                -0.03687145777483,
                -0.02334876845340,
            ],
            [
                -0.47765679842079,
                -0.03687145777483,
                -0.40142945681830,
                -0.00000000000000,
            ],
            [
                -0.03687145777483,
                -0.02334876845340,
                -0.00000000000000,
                -0.07981592633195,
            ],
        ],
        **dd,
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    driver = IntDriver(numbers, GFN1_XTB, ihelp, **dd)
    overlap = Overlap(**dd)
    h0 = GFN1Hamiltonian(numbers, GFN1_XTB, ihelp, cn=None, **dd)

    driver.setup(positions)
    s = overlap.build(driver)

    h = h0.build(positions, s)
    assert pytest.approx(h.cpu(), abs=tol) == h.mT.cpu()
    assert pytest.approx(h.cpu(), abs=tol) == ref.cpu()
