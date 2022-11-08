"""
Test for SCF with charged samples.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

from math import sqrt

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.xtb.calculator import Calculator

from .samples_charged import samples

opts = {"verbosity": 0, "etemp": 300, "guess": "eeq"}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["Ag2Cl22-", "Al3+Ar6", "AD7en+", "C2H4F+", "ZnOOH-"])
def test_single(dtype: torch.dtype, name: str):
    dd = {"dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].item()
    chrg = sample["charge"].type(dtype)

    calc = Calculator(numbers, par, opts=opts, **dd)
    results = calc.singlepoint(numbers, positions, chrg)

    assert pytest.approx(ref, abs=tol, rel=tol) == results.scf.sum(-1).item()
