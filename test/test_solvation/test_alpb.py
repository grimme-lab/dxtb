import pytest
import torch
import math

from xtbml.solvation import alpb
from xtbml.param import GFN1_XTB as par
from xtbml.xtb.calculator import Calculator

from .samples import mb16_43


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sample", [mb16_43["01"], mb16_43["02"]])
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


def test_gb_scf(dtype=torch.float, sample=mb16_43["01"], dielectric_constant=78.9):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["gsolv"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, dtype=dtype)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant)

    calc_vac = Calculator(numbers, positions, par)
    calc_sol = Calculator(numbers, positions, par, interaction=gb)

    results_vac = calc_vac.singlepoint(numbers, positions, charges, verbosity=0)
    results_sol = calc_sol.singlepoint(numbers, positions, charges, verbosity=0)

    gsolv = results_sol["energy"] - results_vac["energy"]

    assert pytest.approx(ref, abs=tol) == gsolv


def test_gb_scf_grad(dtype=torch.float, sample=mb16_43["01"], dielectric_constant=78.9):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).requires_grad_(True)
    ref = sample["gradient"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, dtype=dtype)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant)

    calc = Calculator(numbers, positions, par, interaction=gb)

    results = calc.singlepoint(numbers, positions, charges, verbosity=0)

    energy = results["energy"].sum(-1)
    energy.backward()
    gradient = positions.grad.clone()

    assert pytest.approx(ref, abs=tol) == gradient
