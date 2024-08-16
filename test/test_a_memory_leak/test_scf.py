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
from dxtb import Calculator
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .util import garbage_collect, has_memleak_tensor

opts = {"verbosity": 0, "maxiter": 50, "exclude": ["rep", "disp", "hal"]}
repeats = 5


# FIXME: xitorch's memory leak
@pytest.mark.xfail
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("run_gc", [False, True])
@pytest.mark.parametrize("create_graph", [False, True])
def test_xitorch(dtype: torch.dtype, run_gc: bool, create_graph: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    def fcn():
        sample = samples["LiH"]
        numbers = sample["numbers"].to(DEVICE)
        positions = sample["positions"].clone().to(**dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"scf_mode": "nonpure"})
        calc = Calculator(numbers, par, opts=options, **dd)

        # variables to be differentiated
        positions.requires_grad_(True)

        result = calc.singlepoint(positions, charges)
        energy = result.scf.sum(-1)

        _ = torch.autograd.grad(energy, (positions), create_graph=create_graph)

        # known reference cycle for create_graph=True
        if create_graph is True:
            energy.backward()

        del numbers
        del positions
        del charges
        del calc
        del result
        del energy

    # run garbage collector to avoid leaks across other tests
    garbage_collect()
    leak = has_memleak_tensor(fcn, gccollect=run_gc)
    garbage_collect()

    assert not leak, "Memory leak detected"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("run_gc", [False, True])
@pytest.mark.parametrize("create_graph", [False, True])
def test_xitorch_pure(dtype: torch.dtype, run_gc: bool, create_graph: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    def fcn():
        sample = samples["SiH4"]
        numbers = sample["numbers"].to(DEVICE)
        positions = sample["positions"].clone().to(**dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"scf_mode": "implicit"})
        calc = Calculator(numbers, par, opts=options, **dd)

        # variables to be differentiated
        positions.requires_grad_(True)

        energy = calc.energy(positions, charges)

        _ = torch.autograd.grad(energy, (positions), create_graph=create_graph)

        # known reference cycle for create_graph=True
        if create_graph is True:
            energy.backward()

    # run garbage collector to avoid leaks across other tests
    garbage_collect()
    leak = has_memleak_tensor(fcn, gccollect=run_gc)
    garbage_collect()

    assert not leak, "Memory leak detected"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("run_gc", [True])
# FIXME: not calling the garbage collector also causes a memory leak
# @pytest.mark.parametrize("run_gc", [False, True])
@pytest.mark.parametrize("create_graph", [False, True])
def skip_test_fulltracking(
    dtype: torch.dtype, run_gc: bool, create_graph: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    def fcn():
        sample = samples["SiH4"]
        numbers = sample["numbers"].to(DEVICE)
        positions = sample["positions"].clone().to(**dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
        calc = Calculator(numbers, par, opts=options, **dd)

        # variables to be differentiated
        positions.requires_grad_(True)

        result = calc.singlepoint(positions, charges)
        energy = result.scf.sum(-1)

        _ = torch.autograd.grad(energy, (positions), create_graph=create_graph)

        # known reference cycle for create_graph=True
        if create_graph is True:
            energy.backward()

        del numbers
        del positions
        del charges
        del calc
        del result
        del energy

    # run garbage collector to avoid leaks across other tests
    garbage_collect()
    leak = has_memleak_tensor(fcn, gccollect=run_gc)
    garbage_collect()

    assert not leak, "Memory leak detected"
