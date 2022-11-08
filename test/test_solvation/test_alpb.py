from math import sqrt

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.solvation import alpb
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0, "xitorch_fatol": 1e-10, "xitorch_xatol": 1e-10}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
def test_gb_single(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    dielectric_constant = torch.tensor(dielectric_constant, **dd)

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charges = sample["charges"].type(dtype)
    ref = sample["energies"].type(dtype)

    ihelp = "IndexHelper"
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)
    cache = gb.get_cache(numbers, positions, ihelp)
    energies = gb.get_atom_energy(charges, ihelp, cache)

    assert pytest.approx(energies, abs=tol) == ref


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gb_scf(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = 1e-3 if dtype == torch.float else 1e-5
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gsolv"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, **dd)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)

    calc_vac = Calculator(numbers, par, opts=opts, **dd)
    calc_sol = Calculator(numbers, par, interaction=gb, opts=opts, **dd)

    results_vac = calc_vac.singlepoint(numbers, positions, charges)
    results_sol = calc_sol.singlepoint(numbers, positions, charges)

    gsolv = results_sol.scf - results_vac.scf

    assert pytest.approx(ref, abs=tol) == gsolv


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gb_scf_grad(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = 1e-3 if dtype == torch.float else 1e-5
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    pos = sample["positions"].type(dtype)
    positions = pos.detach().clone().requires_grad_(True)
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
