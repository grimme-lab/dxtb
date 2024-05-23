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
from tad_mctc.autograd import dgradcheck, dgradgradcheck

from dxtb._src.integral.driver.pytorch.impls import md
from dxtb._src.typing import Callable, Tensor

fcoeff_list = [
    md.explicit.ecoeffs_s,
    md.explicit.ecoeffs_p,
    md.explicit.ecoeffs_d,
    md.explicit.ecoeffs_f,
]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype,
    fcoeff,
    l: int,
) -> tuple[Callable[[Tensor, Tensor], Tensor], tuple[Tensor, Tensor]]:
    """Prepare gradient check from `torch.autograd`."""
    lj = torch.tensor(l)
    alpha = (
        torch.tensor(
            [10.256286, 0.622797, 0.239101, 7.611997, 1.392902, 0.386963, 0.128430],
            dtype=dtype,
        ),
        torch.tensor(
            [1.723363, 0.449418, 0.160806, 0.067220, 0.030738, 0.014532],
            dtype=dtype,
        ),
    )

    # variables to be differentiated
    vec = torch.tensor(
        [[-0.000000, -0.000000, -3.015935]],
        dtype=dtype,
        requires_grad=True,
    )

    ai, aj = alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-2)
    eij = ai + aj
    oij = 1.0 / eij
    xij = 0.5 * oij

    rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
    rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

    def func(ri: Tensor, rj: Tensor) -> Tensor:
        return fcoeff(lj, xij, ri, rj)

    return func, (rpi, rpj)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("fcoeff", fcoeff_list)
@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_grad(dtype: torch.dtype, fcoeff, l: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, fcoeff, l)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("fcoeff", fcoeff_list)
@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_gradgrad(dtype: torch.dtype, fcoeff, l: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, fcoeff, l)
    assert dgradgradcheck(func, diffvars, atol=tol)
