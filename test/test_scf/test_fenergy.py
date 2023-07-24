"""
Test energy calculations from SCF iterations.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from .test_elements import uhf_anion, uhf_cation

device = None

opts = {
    "etemp": 300,
    "fermi_maxiter": 500,
    "fermi_thresh": {
        torch.float32: torch.tensor(1e-4, dtype=torch.float32),  # instead of 1e-5
        torch.float64: torch.tensor(1e-10, dtype=torch.float64),
    },
    "use_potential": True,  # important for atoms (better convergence)
    "verbosity": 0,
}


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("partition", ["equal", "atomic"])
def test_element_energy_scf_mode(dtype: torch.dtype, partition: str) -> None:
    """Comparison of object SCF (old) vs. functional SCF."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-8

    def fcn(number, scf_mode):
        numbers = torch.tensor([number])
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(
            opts,
            **{
                "xitorch_fatol": 1e-6,
                "xitorch_xatol": 1e-6,
                "fermi_fenergy_partition": partition,
                "scf_mode": scf_mode,
            },
        )
        calc = Calculator(numbers, par, opts=options, **dd)
        result = calc.singlepoint(numbers, positions, charges)
        return result.scf.sum(-1)

    energies = [fcn(n, "implicit") for n in range(1, 87)]
    energies_old = [fcn(n, "implicit_old") for n in range(1, 87)]
    assert pytest.approx(energies, abs=tol) == energies_old


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("partition", ["equal", "atomic"])
def test_element_scf_mode(dtype: torch.dtype, partition: str) -> None:
    """Comparison of object SCF (old) vs. functional SCF."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-8

    def fcn(number, scf_mode):
        numbers = torch.tensor([number])
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(
            opts,
            **{
                "xitorch_fatol": 1e-6,
                "xitorch_xatol": 1e-6,
                "fermi_fenergy_partition": partition,
                "scf_mode": scf_mode,
            },
        )
        calc = Calculator(numbers, par, opts=options, **dd)
        result = calc.singlepoint(numbers, positions, charges)
        return result.fenergy

    fenergies = [fcn(n, "implicit").item() for n in range(1, 87)]
    fenergies_old = [fcn(n, "implicit_old").item() for n in range(1, 87)]
    assert pytest.approx(fenergies, abs=tol) == fenergies_old


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element(dtype: torch.dtype) -> None:
    """Different free energies for different atoms."""
    dd: DD = {"device": device, "dtype": dtype}

    def fcn(number):
        numbers = torch.tensor([number])
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"xitorch_fatol": 1e-6, "xitorch_xatol": 1e-6})
        calc = Calculator(numbers, par, opts=options, **dd)
        result = calc.singlepoint(numbers, positions, charges)
        return result.fenergy

    fenergies = [fcn(n).item() for n in range(1, 87)]
    unique = set(fenergies)
    assert len(unique) > 5


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_cation(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    def fcn(number):
        numbers = torch.tensor([number])
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(1.0, **dd)

        options = dict(
            opts,
            **{
                "xitorch_fatol": 1e-5,  # avoids Jacobian inversion error
                "xitorch_xatol": 1e-5,  # avoids Jacobian inversion error
                "spin": uhf_cation[number - 1],
            },
        )
        calc = Calculator(numbers, par, opts=options, **dd)
        result = calc.singlepoint(numbers, positions, charges)
        return result.fenergy

    # no (valence) electrons OR gold
    _exclude = [1, 3, 11, 19, 37, 55, 79]
    numbers = [i for i in range(1, 87) if i not in _exclude]

    fenergies = [fcn(n).item() for n in numbers]
    unique = set(fenergies)
    assert len(unique) > 5


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_anion(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    def fcn(number):
        numbers = torch.tensor([number])
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(-1.0, **dd)

        options = dict(
            opts,
            **{
                "xitorch_fatol": 1e-5,  # avoid Jacobian inversion error
                "xitorch_xatol": 1e-5,  # avoid Jacobian inversion error
                "spin": uhf_anion[number - 1],
            },
        )
        calc = Calculator(numbers, par, opts=options, **dd)
        result = calc.singlepoint(numbers, positions, charges)
        return result.fenergy

    # Helium doesn't have enough orbitals for negative charge,
    # SCF does not converge (in tblite too)
    _exclude = [2, 21, 22, 23, 25, 43, 57, 58, 59]
    numbers = [i for i in range(1, 87) if i not in _exclude]

    fenergies = [fcn(n).item() for n in numbers]
    unique = set(fenergies)
    assert len(unique) > 5
