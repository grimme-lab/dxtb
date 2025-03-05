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
Test for self-consistent D4 dispersion energy
Reference values obtained with tblite 0.4.0 disabling all other interactions.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.typing import Tensor
from tad_multicharge import get_eeq_charges

from dxtb import GFN2_XTB
from dxtb._src.typing import DD
from dxtb.components.dispersion import new_d4sc

from ...conftest import DEVICE
from .samples import samples

slist = ["LiH", "SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


@pytest.mark.parametrize("name", slist)
def test_single(name: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = sqrt(torch.finfo(torch.double).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    disp = new_d4sc(numbers, GFN2_XTB, **dd)
    assert disp is not None

    chrg = torch.tensor(0.0, **dd)
    qat = get_eeq_charges(numbers, positions, chrg)

    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        cache = disp.get_cache(numbers=numbers, positions=pos)
        return disp.get_monopole_atom_energy(cache, qat)

    assert dgradcheck(func, positions, atol=tol, nondet_tol=1e-7)


@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_batch(name1: str, name2: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    tol = sqrt(torch.finfo(torch.double).eps)

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

    disp = new_d4sc(numbers, GFN2_XTB, **dd)
    assert disp is not None

    chrg = torch.tensor([0.0, 0.0], **dd)
    qat = get_eeq_charges(numbers, positions, chrg)

    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        cache = disp.get_cache(numbers=numbers, positions=pos)
        return disp.get_monopole_atom_energy(cache, qat)

    assert dgradcheck(func, positions, atol=tol, nondet_tol=1e-7)
