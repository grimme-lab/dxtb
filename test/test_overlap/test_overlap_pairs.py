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
Run tests for overlap of diatomic systems.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb._src.basis import Basis, IndexHelper
from dxtb._src.integral.driver.pytorch.impls.md import overlap_gto
from dxtb._src.typing import DD

from ..utils import load_from_npz
from .samples import samples
from .utils import calc_overlap

ref_overlap = np.load("test/test_overlap/overlap.npz")

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["HC", "HHe", "SCl"])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-05

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, dtype)

    s = calc_overlap(numbers, positions, par, uplo="n", dd=dd)
    assert pytest.approx(ref, rel=tol, abs=tol) == s


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["C", "HC", "HHe", "SCl"])
@pytest.mark.parametrize("name2", ["C", "HC", "HHe", "SCl"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-05

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
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
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        )
    )

    s = calc_overlap(numbers, positions, par, uplo="n", dd=dd)
    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(ref, abs=tol) == s


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_higher_orbitals(dtype: torch.dtype):
    # pylint: disable=import-outside-toplevel
    from .test_cgto_ortho_data import ref_data

    dd: DD = {"device": device, "dtype": dtype}

    vec = torch.tensor([0.0, 0.0, 1.4], **dd)

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, par)
    bas = Basis(number, par, ihelp, **dd)
    alpha, coeff = bas.create_cgtos()

    ai = alpha[0]
    ci = coeff[0]
    aj = alpha[1]
    cj = coeff[1]

    # change momenta artifically for testing purposes
    for i in range(2):
        for j in range(2):
            ref = ref_data[f"{i}-{j}"].to(**dd).T
            overlap = overlap_gto(
                (torch.tensor(i), torch.tensor(j)),
                (ai, aj),
                (ci, cj),
                vec,
            )

            assert pytest.approx(overlap, rel=1e-05, abs=1e-03) == ref
