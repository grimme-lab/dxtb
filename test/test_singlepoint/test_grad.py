"""
Run tests for singlepoint gradient calculation with read from coord file.
"""

from math import sqrt
from pathlib import Path

import pytest
import torch

from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from .samples import samples


# FIXME: Gradients for torch.double only exact up to around 1e-6
@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4", "SiH4", "LYS_xao", "C60"])
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    base = Path(Path(__file__).parent, name)

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    positions.requires_grad_(True)
    charge = torch.tensor(charge).type(dtype)

    ref = samples[name]["grad"].type(dtype)

    calc = Calculator(numbers, positions, par)
    result = calc.singlepoint(numbers, positions, charge, {"verbosity": 0})
    energy = result.total.sum(-1)

    energy.backward()
    if positions.grad is None:
        assert False
    gradient = positions.grad.clone()

    assert torch.allclose(gradient, ref, atol=tol)
