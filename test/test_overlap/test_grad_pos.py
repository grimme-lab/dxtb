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
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb._src.integral.driver.pytorch import OverlapPytorch as Overlap
from dxtb._src.typing import DD, Callable, Tensor

from ..conftest import DEVICE, NONDET_TOL
from .samples import samples

slist = ["LiH", "H2O"]
slist_large = ["H2", "HHe", "Li2", "S2", "SiH4"]

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
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(uplo="n", **dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        driver.setup(p)
        return overlap.build(driver)

    return func, pos


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_grad_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_gradgrad_large(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


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

    # mask needed for un-batched overlap calculation (numerical jacobian in
    # gradcheck also changes the padding values, prohibiting `batch.deflate()`)
    positions, mask = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
        return_mask=True,
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(uplo="n", **dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        driver.setup(p, mask=mask)
        return overlap.build(driver)

    return func, pos


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")  # torch.meshgrid from batch.deflate
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")  # torch.meshgrid from batch.deflate
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", slist_large)
def test_grad_batch_large(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")  # torch.meshgrid from batch.deflate
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")  # torch.meshgrid from batch.deflate
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", slist_large)
def test_gradgrad_batch_large(
    dtype: torch.dtype, name1: str, name2: str
) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=NONDET_TOL)
