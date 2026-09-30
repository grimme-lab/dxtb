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
from tad_mctc.batch import pack

from dxtb import IndexHelper
from dxtb._src.components.classicals import new_repulsion
from dxtb._src.typing import DD, Literal

from ...conftest import DEVICE
from ...utils import get_param_module
from .samples import samples

sample_list = [
    "H2",
    "H2O",
    "SiH4",
    "ZnOOH-",
    "MB16_43_01",
    "MB16_43_02",
    "MB16_43_03",
    "LYS_xao",
]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("par", ["gfn1", "gfn2", "gfn0"])
def test_single(
    dtype: torch.dtype, name: str, par: Literal["gfn1", "gfn2", "gfn0"]
) -> None:
    """Test repulsion calculation for single sample."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample[par].to(**dd)

    _par = get_param_module(par, **dd)

    rep = new_repulsion(torch.unique(numbers), _par, cutoff=50, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, _par)
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache, atom_resolved=False)

    assert pytest.approx(ref.cpu(), abs=tol) == 0.5 * e.sum((-2, -1)).cpu()


# Every sample is already checked individually in `test_single`; the batched
# test only needs to cover packing/padding, so a few representative pairs
# (small+small, small+large, charged, largest, and one identical pair for a
# batch without padding) suffice.
batch_pairs = [
    ("H2", "H2O"),
    ("SiH4", "ZnOOH-"),
    ("MB16_43_01", "H2"),
    ("LYS_xao", "MB16_43_02"),
    ("MB16_43_03", "SiH4"),
    ("SiH4", "SiH4"),
]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1, name2", batch_pairs)
@pytest.mark.parametrize("par", ["gfn1", "gfn2", "gfn0"])
def test_batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    par: Literal["gfn1", "gfn2", "gfn0"],
) -> None:
    """Test repulsion calculation for multiple samples."""
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

    _par = get_param_module(par, **dd)

    rep = new_repulsion(torch.unique(numbers), _par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, _par)
    cache = rep.get_cache(numbers, ihelp)
    e = rep.get_energy(positions, cache)

    assert pytest.approx(ref.cpu(), abs=tol) == e.sum(-1).cpu()
