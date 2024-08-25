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
Test overlap build from integral container.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb import integrals as ints
from dxtb._src.constants.labels import INTDRIVER_ANALYTICAL
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.driver.pytorch import OverlapPytorch
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples


def run(numbers: Tensor, positions: Tensor, dd: DD) -> None:
    ihelp = IndexHelper.from_numbers(numbers, par)
    mgr = DriverManager(INTDRIVER_ANALYTICAL, **dd)
    mgr.create_driver(numbers, par, ihelp)

    i = ints.Integrals(mgr, _overlap=OverlapPytorch(**dd), **dd)
    i.build_overlap(positions)

    o = i.overlap
    assert o is not None
    assert o.matrix is not None


@pytest.mark.parametrize("name", ["H2"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    run(numbers, positions, dd)


@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
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

    run(numbers, positions, dd)
