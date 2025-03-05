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

import pytest
import torch
from tad_mctc import read, read_chrg

from dxtb import GFN2_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import load_from_tblite_grad

f = Path(__file__).resolve().parent / "refs"

opts = {
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "SiH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_backward(dtype: torch.dtype, name: str, scf_mode: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50  # slightly larger for H2O!
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read(Path(base, "coord"), **dd)
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    positions = positions.clone().requires_grad_(True)

    # do calc
    options = dict(
        opts,
        **{
            "mixer": "anderson" if scf_mode == "full" else "broyden",
            "f_atol": 1e-5 if dtype == torch.float else 1e-10,
            "x_atol": 1e-5 if dtype == torch.float else 1e-10,
            "scf_mode": scf_mode,
        },
    )
    calc = Calculator(numbers, GFN2_XTB, opts=options, **dd)
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
    ref = load_from_tblite_grad(f / f"{name.casefold()}.txt", **dd)
    g = ref["gradient"]
    assert pytest.approx(g.cpu(), abs=tol, rel=1e-4) == autograd.cpu()


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4"])
@pytest.mark.parametrize("scf_mode", ["implicit", "nonpure", "full"])
def test_num(name: str, scf_mode: str) -> None:
    dtype = torch.double
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read(Path(base, "coord"), **dd)
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    # do calc
    gradient = num_grad(numbers, positions, charge, scf_mode, dd)

    ref = load_from_tblite_grad(f / f"{name.casefold()}.txt", **dd)
    g = ref["gradient"]
    assert pytest.approx(g.cpu(), abs=1e-5, rel=1e-5) == gradient.cpu()


def num_grad(
    numbers: Tensor, positions: Tensor, charge: Tensor, scf_mode: str, dd: DD
) -> Tensor:
    """Calculate gradient numerically for reference."""

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
            "f_atol": 1e-5 if dd["dtype"] == torch.float else 1e-10,
            "x_atol": 1e-5 if dd["dtype"] == torch.float else 1e-10,
        },
    )
    calc = Calculator(numbers, GFN2_XTB, opts=options, **dd)

    gradient = torch.zeros_like(positions)
    step = 1.0e-5

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
