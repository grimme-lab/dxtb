"""
Run tests for singlepoint calculation with read from coord file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

opts = {"verbosity": 0}


def test_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read_coord(Path("non-existing-coord-file"))


def test_uhf_fail() -> None:
    base = Path(Path(__file__).parent, "mols", "H")

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions)
    charge = torch.tensor(charge)

    calc = Calculator(numbers, par, opts=opts)

    with pytest.raises(ValueError):
        calc.set_option("spin", 0)
        calc.singlepoint(numbers, positions, charge)

    with pytest.raises(ValueError):
        calc.set_option("spin", 2)
        calc.singlepoint(numbers, positions, charge)
