"""
Run tests for singlepoint calculation with read from coord file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.timing import timer
from dxtb.xtb import Calculator

opts = {"verbosity": 0}


def test_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read_coord(Path("non-existing-coord-file"))


def test_uhf_fail() -> None:
    # singlepoint starts SCF timer, but exception is thrown before the SCF
    # timer is stopped, so we must disable it here
    status = timer._enabled
    if status is True:
        timer.disable()

    base = Path(Path(__file__).parent, "mols", "H")

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions)
    charge = torch.tensor(charge)

    calc = Calculator(numbers, par, opts=opts)

    with pytest.raises(ValueError):
        calc.singlepoint(numbers, positions, charge, spin=0)

    with pytest.raises(ValueError):
        calc.singlepoint(numbers, positions, charge, spin=2)

    if status is True:
        timer.enable()
