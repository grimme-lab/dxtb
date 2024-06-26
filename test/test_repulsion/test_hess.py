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
Run tests repulsion Hessian.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.autograd import jacrev
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import new_repulsion
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import reshape_fortran
from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = reshape_fortran(
        sample["gfn1_hess"].to(**dd),
        torch.Size(2 * (numbers.shape[0], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    def energy(pos: Tensor) -> Tensor:
        return rep.get_energy(pos, cache).sum()

    hess = jacrev(jacrev(energy))(positions)
    assert isinstance(hess, Tensor)

    positions.detach_()
    hess = hess.detach().reshape_as(ref)

    assert ref.shape == hess.shape
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

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

    ref = pack(
        [
            reshape_fortran(
                sample1["gfn1_hess"].to(**dd),
                torch.Size(
                    (
                        sample1["numbers"].to(DEVICE).shape[0],
                        3,
                        sample1["numbers"].to(DEVICE).shape[0],
                        3,
                    )
                ),
            ),
            reshape_fortran(
                sample2["gfn1_hess"].to(**dd),
                torch.Size(
                    (sample2["numbers"].shape[0], 3, sample2["numbers"].shape[0], 3)
                ),
            ),
        ]
    )

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    positions.requires_grad_(True)

    def energy(pos: Tensor) -> Tensor:
        return rep.get_energy(pos, cache).sum()

    hess = jacrev(jacrev(energy))(positions)
    assert isinstance(hess, Tensor)

    positions.detach_()
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()
