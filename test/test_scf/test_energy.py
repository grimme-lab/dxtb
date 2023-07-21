
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
def test_element_electronic_free_energy(dtype: torch.dtype) -> None:
    """Different free energies for different atoms."""

    dd: DD = {"device": device, "dtype": dtype}

    def calc(number):
        numbers = torch.tensor([number])
        positions = torch.zeros((1, 3), **dd)
        charges = torch.tensor(0.0, **dd)

        options = dict(opts, **{"xitorch_fatol": 1e-6, "xitorch_xatol": 1e-6})
        calc = Calculator(numbers, par, opts=options, **dd)
        result = calc.singlepoint(numbers, positions, charges)

        return result.fenergy
    
    fenergies = [calc(n).item() for n in range(1, 87)]
    unique = set(fenergies)
    assert len(unique) > 5



@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_cation(dtype: torch.dtype) -> None:
    
    dd: DD = {"device": device, "dtype": dtype}

    def calc(number):
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
    _exclude =  [1, 3, 11, 19, 37, 55, 79] 
    numbers = [i for i in range(1, 87) if i not in _exclude]

    fenergies = [calc(n).item() for n in numbers]
    unique = set(fenergies)
    assert len(unique) > 5


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_anion(dtype: torch.dtype) -> None:
    
    dd: DD = {"device": device, "dtype": dtype}

    def calc(number):
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
    _exclude =  [2, 21, 22, 23, 25, 43, 57, 58, 59]
    numbers = [i for i in range(1, 87) if i not in _exclude]

    fenergies = [calc(n).item() for n in numbers]
    unique = set(fenergies)
    assert len(unique) > 5
