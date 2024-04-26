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
Run tests for memory leak in custom autograd.

Inspired by DQC.
"""

from __future__ import annotations

import gc

import pytest
import torch
from tad_mctc.data.molecules import mols as samples

from dxtb.basis import IndexHelper
from dxtb.components.classicals import new_repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD

from ..utils import nth_derivative
from .util import has_memleak_tensor

sample_list = ["H2O", "SiH4", "MB16_43_01"]

device = None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_single(dtype: torch.dtype, name: str, n: int) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    def fcn():
        assert par.repulsion is not None

        sample = samples[name]
        numbers = sample["numbers"].to(device)
        positions = sample["positions"].clone().to(**dd)

        ihelp = IndexHelper.from_numbers(numbers, par)

        # variables to be differentiated
        positions.requires_grad_(True)

        rep = new_repulsion(numbers, par, **dd)
        assert rep is not None

        cache = rep.get_cache(numbers, ihelp)
        energy = rep.get_energy(positions, cache).sum()

        _ = nth_derivative(energy, positions, n)

    # run garbage collector to avoid leaks across other tests
    gc.collect()
    leak = has_memleak_tensor(fcn)
    gc.collect()

    assert not leak, "Memory leak detected"
