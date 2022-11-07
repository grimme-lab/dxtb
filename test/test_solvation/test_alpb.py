import math

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.solvation import alpb
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sample", [samples["MB16_43_01"], samples["MB16_43_02"]])
def test_gb_single(dtype: torch.dtype, sample, dielectric_constant=78.9):

    dielectric_constant = torch.tensor(dielectric_constant, dtype=dtype)

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charges = sample["charges"].type(dtype)

    ihelp = "IndexHelper"
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant)
    cache = gb.get_cache(numbers, positions, ihelp)
    energies = gb.get_atom_energy(charges, ihelp, cache)
    assert torch.allclose(energies, sample["energies"].type(dtype))


def test_gb_scf(
    dtype=torch.float, sample=samples["MB16_43_01"], dielectric_constant=78.9
):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gsolv"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, dtype=dtype)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant)

    calc_vac = Calculator(numbers, positions, par, opts=opts)
    calc_sol = Calculator(numbers, positions, par, interaction=gb, opts=opts)

    results_vac = calc_vac.singlepoint(numbers, positions, charges)
    results_sol = calc_sol.singlepoint(numbers, positions, charges)

    gsolv = results_sol.scf - results_vac.scf

    assert pytest.approx(ref, abs=tol) == gsolv


def test_gb_scf_grad(
    dtype=torch.float, sample=samples["MB16_43_01"], dielectric_constant=78.9
):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).requires_grad_(True)
    ref = sample["gradient"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, dtype=dtype)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant)

    calc = Calculator(numbers, positions, par, interaction=gb, opts=opts)

    results = calc.singlepoint(numbers, positions, charges)
    energy = results.scf.sum(-1)
    energy.backward()

    if positions.grad is None:
        assert False
    gradient = positions.grad.clone()

    assert pytest.approx(ref, abs=tol) == gradient
