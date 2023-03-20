"""
Test gradient of the Analytical linearized Poisson-Boltzmann model.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.solvation import alpb
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0, "xitorch_fatol": 1e-10, "xitorch_xatol": 1e-10}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gb_scf_grad(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = 1e-3 if dtype == torch.float else 1e-5
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)
    ref = sample["gradient"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, **dd)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)

    calc = Calculator(numbers, par, interaction=gb, opts=opts, **dd)

    results = calc.singlepoint(numbers, positions, charges)
    energy = results.scf.sum(-1)
    energy.backward()

    if positions.grad is None:
        assert False
    gradient = positions.grad.clone()

    assert pytest.approx(ref, abs=tol) == gradient

    positions.detach_()
