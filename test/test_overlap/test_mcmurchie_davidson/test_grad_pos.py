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
from dxtb._src.integral.driver.pytorch.impls.md import recursion
from dxtb._src.typing import Callable, Tensor, DD
from ...conftest import DEVICE

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, md_func, angular: tuple[Tensor, Tensor]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    alpha = (
        torch.tensor(
            [10.256286, 0.622797, 0.239101, 7.611997, 1.392902, 0.386963, 0.128430],
            **dd,
        ),
        torch.tensor(
            [1.723363, 0.449418, 0.160806, 0.067220, 0.030738, 0.014532],
            **dd,
        ),
    )
    coeff = (
        torch.tensor(
            [-1.318654, 1.603878, 0.601323, -0.980904, -1.257964, -0.985990, -0.235962],
            **dd,
        ),
        torch.tensor(
            [0.022303, 0.026981, 0.027555, 0.019758, 0.007361, 0.000756],
            **dd,
        ),
    )

    # variables to be differentiated
    vec = torch.tensor(
        [[-0.000000, -0.000000, -3.015935]],
        **dd,
        requires_grad=True,
    )

    def func(v: Tensor) -> Tensor:
        return md_func(angular, alpha, coeff, v)

    return func, vec


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("md_func", [md.explicit.md_explicit, recursion.md_recursion])
@pytest.mark.parametrize("li", [0, 1, 2, 3])
@pytest.mark.parametrize("lj", [0, 1, 2, 3])
def test_grad(dtype: torch.dtype, md_func, li: int, lj: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    angular = (torch.tensor(li), torch.tensor(lj))
    func, diffvars = gradchecker(dtype, md_func, angular)
    assert dgradcheck(func, diffvars, atol=tol)


# NOTE: Recursive version fails because of inplace operations
@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("md_func", [md.explicit.md_explicit])
@pytest.mark.parametrize("li", [0, 1, 2, 3])
@pytest.mark.parametrize("lj", [0, 1, 2, 3])
def test_gradgrad(dtype: torch.dtype, md_func, li: int, lj: int) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    angular = (torch.tensor(li), torch.tensor(lj))
    func, diffvars = gradchecker(dtype, md_func, angular)
    assert dgradgradcheck(func, diffvars, atol=tol)
