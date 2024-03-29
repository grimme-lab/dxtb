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
Run tests for Hessian of halogen bond correction.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.components.classicals import new_halogen
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch, hessian

from ..utils import reshape_fortran
from .samples import samples

sample_list = ["br2nh3", "br2och2", "finch", "LiH", "SiH4", "MB16_43_01"]

device = None


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    # hessian
    hess = hessian(xb.get_energy, (positions, cache))
    positions.detach_()
    hess = hess.detach().reshape_as(ref)

    assert ref.shape == hess.shape
    assert pytest.approx(ref, abs=tol, rel=tol) == hess


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["finch"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    ref = batch.pack(
        [
            reshape_fortran(
                sample1["hessian"].to(**dd),
                torch.Size(2 * (sample1["numbers"].to(device).shape[-1], 3)),
            ),
            reshape_fortran(
                sample2["hessian"].to(**dd),
                torch.Size(2 * (sample2["numbers"].shape[-1], 3)),
            ),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    hess = hessian(xb.get_energy, (positions, cache), is_batched=True)
    # print(hess)
    # print(ref_hess)
    # print(hess.shape)

    assert pytest.approx(ref, abs=tol, rel=tol) == hess.detach()

    positions.detach_()
