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

import pytest
import torch
from tad_mctc.data.molecules import mols as samples

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import Repulsion
from dxtb._src.param import get_elem_param
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .util import garbage_collect, has_memleak_tensor

slist = ["H2O", "SiH4"]
slist_large = ["MB16_43_01"]


def execute(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    def fcn():
        assert par.repulsion is not None

        sample = samples[name]
        numbers = sample["numbers"].to(DEVICE)
        positions = sample["positions"].clone().to(**dd)

        ihelp = IndexHelper.from_numbers(numbers, par)

        # variables to be differentiated
        arep = get_elem_param(
            torch.unique(numbers),
            par.element,
            "arep",
            pad_val=0,
            **dd,
            requires_grad=True,
        )
        zeff = get_elem_param(
            torch.unique(numbers),
            par.element,
            "zeff",
            pad_val=0,
            **dd,
            requires_grad=True,
        )
        kexp = torch.tensor(
            par.repulsion.effective.kexp,
            **dd,
            requires_grad=True,
        )
        pos = positions.clone().requires_grad_(True)

        rep = Repulsion(arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)

        energy = rep.get_energy(pos, cache).sum()
        _ = torch.autograd.grad(
            energy, (pos, arep, zeff, kexp), create_graph=True
        )

        # known reference cycle for create_graph=True
        energy.backward()

        del numbers
        del pos
        del ihelp
        del rep
        del cache
        del arep
        del zeff
        del kexp

    # run garbage collector to avoid leaks across other tests
    garbage_collect()
    leak = has_memleak_tensor(fcn)
    garbage_collect()

    assert not leak, "Memory leak detected"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    execute(name, dtype)


@pytest.mark.large
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_large(dtype: torch.dtype, name: str) -> None:
    execute(name, dtype)
