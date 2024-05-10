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
Run tests for overlap of molecules.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb import GFN1_XTB as par
from dxtb._src.typing import DD
from dxtb._src.utils import batch
from dxtb.integrals.wrappers import overlap
from dxtb.labels import INTDRIVER_ANALYTICAL

from ..utils import load_from_npz
from .samples import samples
from .utils import calc_overlap

ref_overlap = np.load("test/test_overlap/overlap.npz")

molecules = ["H2O", "CH4", "SiH4", "PbH4-BiH3"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", molecules)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, dtype)

    # more granular overlap calculation
    s = calc_overlap(numbers, positions, par, uplo="n", dd=dd)

    # convenience wrapper with defaults (but pytorch driver)
    s2 = overlap(numbers, positions, par, driver=INTDRIVER_ANALYTICAL)

    assert pytest.approx(ref, abs=tol) == s
    assert pytest.approx(ref, abs=tol) == s2
    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(s, abs=tol) == s2


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", molecules)
@pytest.mark.parametrize("name2", molecules)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        )
    )

    s = calc_overlap(numbers, positions, par, uplo="n", dd=dd)

    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(s, abs=tol) == ref
