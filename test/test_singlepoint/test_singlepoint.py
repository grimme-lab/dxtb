"""
Run tests for singlepoint calculation with read from coord file.
"""

from math import sqrt
from pathlib import Path

import pytest
import torch

from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "name", ["H2", "H2O", "CH4", "SiH4", "LYS_xao", "C60", "vancoh2"]
)
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    base = Path(Path(__file__).parent, name)

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    charge = torch.tensor(charge).type(dtype)

    ref = samples[name]["etot"].item()

    calc = Calculator(numbers, positions, par)
    result = calc.singlepoint(numbers, positions, charge, {"verbosity": 0})
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1).item()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "H2O"])
@pytest.mark.parametrize("name2", ["H2", "CH4"])
@pytest.mark.parametrize("name3", ["H2", "SiH4", "LYS_xao"])
def test_batch(dtype: torch.dtype, name1: str, name2: str, name3: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    numbers, positions, charge = [], [], []
    for name in [name1, name2, name3]:
        base = Path(Path(__file__).parent, name)

        nums, pos = read_coord(Path(base, "coord"))
        chrg = read_chrg(Path(base, ".CHRG"))

        numbers.append(torch.tensor(nums, dtype=torch.long))
        positions.append(torch.tensor(pos).type(dtype))
        charge.append(torch.tensor(chrg).type(dtype))

    numbers = batch.pack(numbers)
    positions = batch.pack(positions)
    charge = batch.pack(charge)
    ref = batch.pack(
        [
            samples[name1]["etot"].type(dtype),
            samples[name2]["etot"].type(dtype),
            samples[name3]["etot"].type(dtype),
        ]
    )

    calc = Calculator(numbers, positions, par)
    result = calc.singlepoint(numbers, positions, charge, {"verbosity": 0})
    assert torch.allclose(ref, result.total.sum(-1), atol=tol, rtol=tol)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H", "NO2"])
def test_uhf_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    base = Path(Path(__file__).parent, name)

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions).type(dtype)
    charge = torch.tensor(charge).type(dtype)

    calc = Calculator(numbers, positions, par)

    ref = samples[name]["etot"].item()
    result = calc.singlepoint(numbers, positions, charge, {"verbosity": 0})
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1).item()


def test_fail() -> None:
    with pytest.raises(FileNotFoundError):
        read_coord(Path("non-existing-coord-file"))


def test_uhf_fail() -> None:
    base = Path(Path(__file__).parent, "H")

    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions)
    charge = torch.tensor(charge)

    calc = Calculator(numbers, positions, par)

    with pytest.raises(ValueError):
        calc.singlepoint(numbers, positions, charge, {"spin": 0})

    with pytest.raises(ValueError):
        calc.singlepoint(numbers, positions, charge, {"spin": 2})
