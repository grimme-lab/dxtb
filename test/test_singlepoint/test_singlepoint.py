"""
Run tests for singlepoint calculation with read from coord file.
"""

from __future__ import annotations
from math import sqrt
from pathlib import Path
import pytest
import torch

from xtbml.exlibs.tbmalt import batch
from xtbml.io import read_chrg, read_coord
from xtbml.param import GFN1_XTB as par
from xtbml.xtb.calculator import Calculator

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "SiH4", "LYS_xao", "C60", "vancoh2"])
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    base = Path(Path(__file__).parent, name)

    geo = read_coord(Path(base, "coord"))
    numbers = [g[-1] for g in geo]
    positions = [g[:3] for g in geo]

    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    charge = torch.tensor(charge).type(dtype)

    calc = Calculator(numbers, positions, par)

    ref = samples[name]["etot"].item()
    result = calc.singlepoint(numbers, positions, charge, verbosity=0)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1).item()
