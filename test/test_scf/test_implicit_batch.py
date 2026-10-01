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
Implicit SCF on a batch in which one system receives no gradient (e.g., one
row of a batched Jacobian such as the force Hessian).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols

from dxtb import GFN1_XTB, Calculator, OutputHandler


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_gradient_for_one_system(dtype: torch.dtype) -> None:
    """Gradient of the charges of system 0 only: no error, no dense solve."""
    dd = {"device": torch.device("cpu"), "dtype": dtype}
    numbers = pack([mols["LiH"]["numbers"], mols["H2O"]["numbers"]])
    positions = pack(
        [mols["LiH"]["positions"].to(**dd), mols["H2O"]["positions"].to(**dd)]
    ).requires_grad_(True)

    opts = {"scf_mode": "implicit", "verbosity": 0}
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    q = calc.singlepoint(positions).charges.mono

    OutputHandler.clear_warnings()
    (grad,) = torch.autograd.grad(q[0].square().sum(), positions)
    assert not any("dense" in msg for msg, _ in OutputHandler.warnings)

    assert torch.isfinite(grad).all()
    assert (grad[1] == 0).all()
    assert grad[0].abs().max() > 0
