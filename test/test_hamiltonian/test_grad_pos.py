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

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.integral.container import Overlap
from dxtb._src.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb._src.ncoord import cn_d3
from dxtb._src.typing import DD, Callable, Tensor
from tad_mctc.batch import pack
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "S2", "H2O", "SiH4"]

tol = 1e-7


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    overlap = Overlap(driver=labels.INTDRIVER_ANALYTICAL, **dd)

    driver = IntDriver(numbers, par, ihelp, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        driver.setup(positions)
        s = overlap.build(driver)
        cn = cn_d3(numbers, pos)
        return h0.build(pos, s, cn=cn)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_grad_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=True)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gradgrad_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=True)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions, mask = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
        return_mask=True,
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    overlap = Overlap(driver=labels.INTDRIVER_ANALYTICAL, **dd)

    driver = IntDriver(numbers, par, ihelp, **dd)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        driver.setup(positions, mask=mask)
        s = overlap.build(driver)
        cn = cn_d3(numbers, pos)
        return h0.build(pos, s, cn=cn)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=1e-7)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=1e-7)
