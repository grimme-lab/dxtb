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
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float`.)
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb import IndexHelper
from dxtb._src.components.classicals import new_repulsion
from dxtb._src.param.gfn1 import GFN1_XTB
from dxtb._src.param.gfn2 import GFN2_XTB
from dxtb._src.typing import DD, Literal
from tad_mctc.batch import pack

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "H2O", "SiH4", "ZnOOH-", "MB16_43_01", "MB16_43_02", "LYS_xao"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("par", ["gfn1", "gfn2"])
def test_single(dtype: torch.dtype, name: str, par: Literal["gfn1", "gfn2"]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[par].to(**dd)

    if par == "gfn1":
        _par = GFN1_XTB
    elif par == "gfn2":
        _par = GFN2_XTB
    else:
        assert False

    rep = new_repulsion(numbers, _par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, _par)
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache, atom_resolved=False)

    assert pytest.approx(ref.cpu(), abs=tol) == 0.5 * e.sum((-2, -1)).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("par", ["gfn1", "gfn2"])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, par: Literal["gfn1", "gfn2"]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

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
            sample1[par].to(**dd),
            sample2[par].to(**dd),
        ],
    )

    if par == "gfn1":
        _par = GFN1_XTB
    elif par == "gfn2":
        _par = GFN2_XTB
    else:
        assert False

    rep = new_repulsion(numbers, _par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, _par)
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref.cpu(), abs=tol) == e.sum(-1).cpu()
