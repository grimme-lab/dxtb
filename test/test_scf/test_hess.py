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
Run tests for SCF Hessian.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import jacrev
from tad_mctc.convert import reshape_fortran

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["LiH", "SiH4"]

opts = {
    "exclude": ["disp", "hal", "rep"],
    "int_driver": "dxtb",
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "f_atol": 1.0e-12,
    "x_atol": 1.0e-12,
    "verbosity": 0,
}


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    atol, rtol = 1e-4, 1e-1  # should be lower!

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = reshape_fortran(
        samples[name]["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    calc = Calculator(numbers, par, opts=opts, **dd)

    pos = positions.clone()

    # numerical hessian
    numref = _numhess(calc, numbers, pos, charge)

    # variable to be differentiated
    pos.requires_grad_(True)

    def energy(pos: Tensor) -> Tensor:
        result = calc.singlepoint(pos, charge)
        return result.scf.sum()

    hess = jacrev(jacrev(energy))(pos)
    assert isinstance(hess, Tensor)

    pos.detach_()
    hess = hess.detach().reshape_as(ref)
    numref = numref.reshape_as(ref)

    assert ref.shape == numref.shape == hess.shape
    assert pytest.approx(ref.cpu(), abs=1e-6, rel=1e-6) == numref.cpu()
    assert pytest.approx(ref.cpu(), abs=atol, rel=rtol) == hess.cpu()


def _numhess(
    calc: Calculator, numbers: Tensor, positions: Tensor, charge: Tensor
) -> Tensor:
    """Calculate numerical Hessian for reference."""

    hess = torch.zeros(
        *(*positions.shape, *positions.shape),
        **{"device": positions.device, "dtype": positions.dtype},
    )

    def _gradfcn(pos: Tensor, charge: Tensor) -> Tensor:
        pos.requires_grad_(True)
        result = -calc.forces_analytical(pos, charge)
        pos.detach_()
        return result.detach()

    step = 1.0e-5
    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            gr = _gradfcn(positions, charge)

            positions[i, j] -= 2 * step
            gl = _gradfcn(positions, charge)

            positions[i, j] += step
            hess[:, :, i, j] = 0.5 * (gr - gl) / step

    return hess
