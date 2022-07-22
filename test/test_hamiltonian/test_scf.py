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
@pytest.mark.parametrize("name", ["H2_cn", "LiH", "SiH4_cn"])
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
@pytest.mark.parametrize("name", ["S2", "PbH4-BiH3", "C6H5I-CH3SH"])
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
