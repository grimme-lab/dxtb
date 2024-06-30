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
from tad_mctc.autograd import jacrev
from tad_mctc.convert import reshape_fortran
from tad_mctc.io import read

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from ..test_dispersion.samples import samples as samples_disp
from ..test_halogen.samples import samples as samples_hal
from ..test_repulsion.samples import samples as samples_rep

opts = {
    "f_atol": 1.0e-10,
    "x_atol": 1.0e-10,
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

sample_list = ["LiH", "SiH4", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    numbers, positions = read.read_from_path(Path(base, "coord"))
    charge = read.read_chrg_from_path(Path(base, ".CHRG"))

    # convert to tensors
    numbers = torch.tensor(numbers, dtype=torch.long, device=DEVICE)
    positions = torch.tensor(positions, **dd)
    charge = torch.tensor(charge, **dd)

    ref = reshape_fortran(
        (
            samples_disp[name]["hessian"]
            + samples_hal[name]["hessian"]
            + samples_rep[name]["gfn1_hess"]
        ).to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    options = dict(opts, **{"exclude": ["scf"]})
    calc = Calculator(numbers, par, opts=options, **dd)

    def energy(pos: Tensor) -> Tensor:
        result = calc.singlepoint(pos, charge)
        return result.total.sum()

    hess = jacrev(jacrev(energy))(positions)
    assert isinstance(hess, Tensor)

    positions.detach_()
    hess = hess.detach().reshape_as(ref)

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.cpu()
