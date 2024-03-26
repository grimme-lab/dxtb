"""
Run tests for memory leak in custom autograd.

Inspired by DQC.
"""

from __future__ import annotations

import gc

import pytest
import torch
from tad_mctc.data.molecules import mols as samples
from tad_mctc.typing import DD

from dxtb.basis import IndexHelper
from dxtb.classical import Repulsion
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular, get_elem_param

from .util import has_memleak_tensor

sample_list = ["H2O", "SiH4", "MB16_43_01"]

device = None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    def fcn():
        assert par.repulsion is not None

        sample = samples[name]
        numbers = sample["numbers"].to(device)
        positions = sample["positions"].clone().to(**dd)

        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

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
        positions.requires_grad_(True)

        rep = Repulsion(arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)

        energy = rep.get_energy(positions, cache).sum()
        _ = torch.autograd.grad(
            energy, (positions, arep, zeff, kexp), create_graph=True
        )

        # known reference cycle for create_graph=True
        energy.backward()

    # run garbage collector to avoid leaks across other tests
    gc.collect()
    leak = has_memleak_tensor(fcn)
    gc.collect()

    assert not leak, "Memory leak detected"
