"""
Test for SCF with charged samples.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

import math
import pytest
import torch

from xtbml.param import GFN1_XTB as par
from xtbml.xtb.calculator import Calculator

from .samples_charged import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["Ag2Cl22-", "Al3+Ar6", "AD7en+", "ZnOOH-"])
def test_single(dtype: torch.dtype, name: str):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].item()
    charges = sample["charge"].type(dtype)

    calc = Calculator(numbers, positions, par)

    results = calc.singlepoint(numbers, positions, charges, verbosity=0)
    assert pytest.approx(ref, abs=tol) == results.scf.sum(-1).item()
