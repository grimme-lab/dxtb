"""
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

import math
import pytest
import torch

from xtbml.param import GFN1_XTB as par
from xtbml.xtb.calculator import Calculator

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C", "H2", "LiH", "SiH4"])
def test_single(dtype: torch.dtype, name: str):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"]
    charges = torch.tensor(0.0).type(dtype)

    calc = Calculator(numbers, positions, par)

    results = calc.singlepoint(numbers, positions, charges, verbosity=0)
    assert pytest.approx(ref, abs=tol) == results["energy"].sum(-1).item()


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize(
    "name", ["S2", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01", "LYS_xao", "C60"]
)
def test_single2(dtype: torch.dtype, name: str):
    """Test a few larger system (only float32 as they take some time)."""
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"]
    charges = torch.tensor(0.0).type(dtype)

    calc = Calculator(numbers, positions, par)

    results = calc.singlepoint(numbers, positions, charges, verbosity=0)
    assert pytest.approx(ref, abs=tol) == results["energy"].sum(-1).item()


@pytest.mark.parametrize(
    "testcase",
    [
        # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
        (
            "LiH",
            torch.tensor(
                [
                    [0.0, 0.0, -1.9003812730202383e-2],
                    [0.0, 0.0, +1.9003812730202383e-2],
                ]
            ),
        ),
    ],
)
def test_grad(testcase, dtype: torch.dtype = torch.float):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    name, ref = testcase
    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].type(dtype).requires_grad_(True)
    charges = torch.tensor(0.0, dtype=dtype)

    calc = Calculator(numbers, positions, par)

    results = calc.singlepoint(numbers, positions, charges, verbosity=0)
    energy = results["energy"].sum(-1)

    energy.backward()
    gradient = positions.grad.clone()
    assert torch.allclose(gradient, ref, atol=tol)
