"""
Run tests for memory leak in custom autograd.

Inspired by DQC.
"""
from __future__ import annotations

import gc

import pytest
import torch

from dxtb._types import DD
from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from ..molecules import mols as samples
from .util import has_memleak_tensor

opts = {"verbosity": 0, "maxiter": 50, "exclude": ["rep", "disp", "hal"]}

device = None


# FIXME: xitorch's memory leak
@pytest.mark.xfail
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("run_gc", [False, True])
@pytest.mark.parametrize("create_graph", [False, True])
def test_xitorch(dtype: torch.dtype, run_gc: bool, create_graph: bool) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    def fcn():
        assert par.repulsion is not None

        sample = samples["SiH4"]
        numbers = sample["numbers"].to(device)
        positions = sample["positions"].clone().to(**dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"scf_mode": "implicit"})
        calc = Calculator(numbers, par, opts=options, **dd)

        # variables to be differentiated
        positions.requires_grad_(True)

        result = calc.singlepoint(numbers, positions, charges)
        energy = result.scf.sum(-1)

        _ = torch.autograd.grad(energy, (positions), create_graph=create_graph)

        # known reference cycle for create_graph=True
        if create_graph is True:
            energy.backward()

    assert not has_memleak_tensor(fcn, gccollect=run_gc), "Memory leak detected"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("run_gc", [True])
# FIXME: not calling the garbage collector also causes a memory leak
# @pytest.mark.parametrize("run_gc", [False, True])
@pytest.mark.parametrize("create_graph", [False, True])
def test_fulltracking(dtype: torch.dtype, run_gc: bool, create_graph: bool) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    def fcn():
        sample = samples["SiH4"]
        numbers = sample["numbers"].to(device)
        positions = sample["positions"].clone().to(**dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
        calc = Calculator(numbers, par, opts=options, **dd)

        # variables to be differentiated
        positions.requires_grad_(True)

        result = calc.singlepoint(numbers, positions, charges)
        energy = result.scf.sum(-1)

        _ = torch.autograd.grad(energy, (positions), create_graph=create_graph)

        # known reference cycle for create_graph=True
        if create_graph is True:
            energy.backward()

    leak = has_memleak_tensor(fcn, gccollect=run_gc)

    # run garbage collector to avoid leaks across other tests
    gc.collect()

    assert not leak, "Memory leak detected"
