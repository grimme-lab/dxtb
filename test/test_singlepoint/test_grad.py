"""
Run tests for singlepoint gradient calculation with read from coord file.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import torch

from dxtb._types import Any, Tensor
from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator

from ..utils import load_from_npz

ref_grad = np.load("test/test_singlepoint/grad.npz")
"""['H2', 'H2O', 'CH4', 'SiH4', 'LYS_xao', 'AD7en+', 'C60', 'vancoh2']"""

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "xitorch_fatol": 1.0e-10,
    "xitorch_xatol": 1.0e-10,
}


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4"])
def test_analytical(dtype: torch.dtype, name: str) -> None:
    atol, rtol = 1e-6, 1e-4
    analytical(dtype, name, atol, rtol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60"])
def test_analytical_large(dtype: torch.dtype, name: str) -> None:
    atol, rtol = 1e-6, 1e-4
    analytical(dtype, name, atol, rtol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["AD7en+", "LYS_xao"])
def test_analytical_large2(dtype: torch.dtype, name: str) -> None:
    atol, rtol = 1e-5, 1e-3
    analytical(dtype, name, atol, rtol)


def analytical(dtype: torch.dtype, name: str, atol: float, rtol: float) -> None:
    dd = {"dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd, requires_grad=True)
    charge = torch.tensor(charge, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, charge, grad=True)
    gradient = result.total_grad.detach()

    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(gradient, abs=atol, rel=rtol) == ref


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "SiH4"])
def test_backward(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50  # slightly larger for H2O!
    dd = {"dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd, requires_grad=True)
    charge = torch.tensor(charge, **dd)

    # do calc
    calc = Calculator(numbers, par, opts=opts, **dd)
    result = calc.singlepoint(numbers, positions, charge)
    energy = result.total.sum(-1)

    # autograd
    energy.backward()
    if positions.grad is None:
        assert False
    autograd = positions.grad.clone()

    # tblite reference grad
    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(autograd, abs=tol, rel=1e-4) == ref


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4"])
def test_num(name: str) -> None:
    dtype = torch.double
    dd = {"dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd)
    charge = torch.tensor(charge, **dd)

    # do calc
    gradient = calc_numerical_gradient(numbers, positions, charge, dd)

    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(gradient, abs=1e-6, rel=1e-4) == ref


def calc_numerical_gradient(
    numbers: Tensor, positions: Tensor, charge: Tensor, dd: dict[str, Any]
) -> Tensor:
    """Calculate gradient numerically for reference."""

    calc = Calculator(numbers, par, opts=opts, **dd)

    gradient = torch.zeros_like(positions)
    step = 1.0e-6

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            result = calc.singlepoint(numbers, positions, charge)
            er = result.total.sum(-1)

            positions[i, j] -= 2 * step
            result = calc.singlepoint(numbers, positions, charge)
            el = result.total.sum(-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
