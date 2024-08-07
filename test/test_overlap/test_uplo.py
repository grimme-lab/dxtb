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
Run tests for overlap of atoms.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..utils import load_from_npz
from .samples import samples
from .utils import calc_overlap

ref_overlap = np.load("test/test_overlap/overlap.npz")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "C", "LiH", "SiH4"])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1e-05

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, dtype)

    s = calc_overlap(numbers, positions, par, dd, uplo="n")
    s_lower = calc_overlap(numbers, positions, par, dd, uplo="l")
    s_upper = calc_overlap(numbers, positions, par, dd, uplo="u")

    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == s.cpu()
    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == s_lower.cpu()
    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == s_upper.cpu()
    assert pytest.approx(s.cpu(), rel=tol, abs=tol) == s_lower.cpu()
    assert pytest.approx(s.cpu(), rel=tol, abs=tol) == s_upper.cpu()


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["C", "Rn"])
@pytest.mark.parametrize("name2", ["C", "Rn"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ),
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        )
    )

    s = calc_overlap(numbers, positions, par, dd, uplo="n")
    s_lower = calc_overlap(numbers, positions, par, dd, uplo="l")
    s_upper = calc_overlap(numbers, positions, par, dd, uplo="u")

    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == s.cpu()
    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == s_lower.cpu()
    assert pytest.approx(ref.cpu(), rel=tol, abs=tol) == s_upper.cpu()
    assert pytest.approx(s.cpu(), rel=tol, abs=tol) == s_lower.cpu()
    assert pytest.approx(s.cpu(), rel=tol, abs=tol) == s_upper.cpu()
