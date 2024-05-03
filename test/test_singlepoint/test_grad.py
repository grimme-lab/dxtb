# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Run tests for singlepoint gradient calculation with read from coord file.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import torch

from dxtb.constants import labels
from dxtb.io import read_chrg, read_coord
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD, Tensor
from dxtb.xtb import Calculator

from ..utils import load_from_npz

ref_grad = np.load("test/test_singlepoint/grad.npz")
"""['H2', 'H2O', 'CH4', 'SiH4', 'LYS_xao', 'AD7en+', 'C60', 'vancoh2']"""

opts = {
    "f_atol": 1.0e-10,
    "x_atol": 1.0e-10,
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

device = None


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_analytical(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    atol, rtol = 1e-5, 1e-4
    analytical(dtype, name, atol, rtol, scf_mode)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_analytical_large(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    atol = rtol = sqrt(torch.finfo(dtype).eps)
    analytical(dtype, name, atol, rtol, scf_mode)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["AD7en+", "LYS_xao"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_analytical_large2(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    atol, rtol = 1e-5, 1e-3
    analytical(dtype, name, atol, rtol, scf_mode)


def analytical(
    dtype: torch.dtype, name: str, atol: float, rtol: float, scf_mode: str
) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd, requires_grad=True)
    charge = torch.tensor(charge, **dd)

    calc = Calculator(numbers, par, opts=opts, **dd)
    result = -calc.forces_analytical(positions, charge)
    gradient = result.detach()

    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(ref, abs=atol, rel=rtol) == gradient


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "SiH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_backward(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50  # slightly larger for H2O!
    dd: DD = {"device": device, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd, requires_grad=True)
    charge = torch.tensor(charge, **dd)

    # do calc
    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)
    result = calc.singlepoint(positions, charge)
    energy = result.total.sum(-1)

    # autograd
    energy.backward()
    assert positions.grad is not None
    autograd = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    # tblite reference grad
    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(ref, abs=tol, rel=1e-4) == autograd


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_num(name: str, scf_mode: str) -> None:
    dtype = torch.double
    dd: DD = {"device": device, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read_coord(Path(base, "coord"))
    charge = read_chrg(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long)
    positions = torch.tensor(positions, **dd)
    charge = torch.tensor(charge, **dd)

    # do calc
    gradient = calc_numerical_gradient(numbers, positions, charge, scf_mode, dd)

    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(ref, abs=1e-6, rel=1e-4) == gradient


def calc_numerical_gradient(
    numbers: Tensor, positions: Tensor, charge: Tensor, scf_mode: str, dd: DD
) -> Tensor:
    """Calculate gradient numerically for reference."""

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    gradient = torch.zeros_like(positions)
    step = 1.0e-6

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            result = calc.singlepoint(positions, charge)
            er = result.total.sum(-1)

            positions[i, j] -= 2 * step
            result = calc.singlepoint(positions, charge)
            el = result.total.sum(-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
