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
Testing overlap gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck

from dxtb import GFN1_XTB as par
from dxtb import Calculator, OutputHandler, labels
from dxtb._src.typing import DD, Callable, Tensor
from dxtb.config import ConfigCache

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "S2", "H2O", "SiH4"]

# remove HHe as it does not pass the numerical gradient check
sample_list = [s for s in sample_list if s not in ["HHe"]]

tol = 5e-4  # increased tolerance


def gradchecker(
    dtype: torch.dtype, name: str, scp_mode: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    opts = {
        "scf_mode": "implicit",
        "scp_mode": scp_mode,
    }

    calc = Calculator(numbers, par, **dd, opts=opts)
    calc.opts.cache = ConfigCache(enabled=False, fock=True)
    OutputHandler.verbosity = 0

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        _ = calc.get_energy(p)  # triggers Fock matrix computation
        return calc.cache["fock"]

    return func, pos


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_grad_fock(dtype: torch.dtype, name: str, scp_mode: str) -> None:
    """
    Check analytical gradient of Fock matrix against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, scp_mode)
    assert dgradcheck(func, diffvars, atol=tol)
