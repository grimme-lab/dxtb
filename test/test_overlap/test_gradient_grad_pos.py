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
Testing autodiff for analytical overlap gradient.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.impls import overlap_gradient
from dxtb._src.typing import DD, Callable, Literal, Tensor

from .samples import samples

sample_list = ["H2", "HHe", "LiH", "SiH4"]

tol = 1e-7

from ..conftest import DEVICE


def gradchecker(
    dtype: torch.dtype, name: str, uplo: Literal["l", "n"]
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(torch.unique(numbers), par, ihelp, **dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        return overlap_gradient(p, bas, ihelp, uplo=uplo)

    return func, pos


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("uplo", ["l", "n"])
def test_grad(dtype: torch.dtype, name: str, uplo: Literal["l", "n"]) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, uplo)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("uplo", ["l", "n"])
def test_gradgrad(
    dtype: torch.dtype, name: str, uplo: Literal["l", "n"]
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, uplo)
    assert dgradgradcheck(func, diffvars, atol=tol)
