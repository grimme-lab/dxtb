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
Testing autodiff gradient of various integrals and their derivatives.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs import libcint
from dxtb._src.integral.driver.libcint import IntDriverLibcint, OverlapLibcint
from dxtb._src.typing import DD, Callable, Tensor
from dxtb._src.utils import is_basis_list

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "Li2", "S2", "H2O", "SiH4"]
int_list = ["ovlp", "r0", "r0r0"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str, intstr: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        atombases = bas.create_libcint(pos)
        assert is_basis_list(atombases)

        wrapper = libcint.LibcintWrapper(atombases, ihelp, spherical=False)
        return libcint.int1e(intstr, wrapper)

    return func, pos


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("intstr", int_list)
@pytest.mark.parametrize("deriv", ["", "ip"])
def test_grad(dtype: torch.dtype, name: str, intstr: str, deriv: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, deriv + intstr)
    assert dgradcheck(func, diffvars, atol=tol, rtol=tol, nondet_tol=1e-7)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("intstr", int_list)
def test_gradgrad(dtype: torch.dtype, name: str, intstr: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, intstr)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=1e-7)


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
    driver = IntDriverLibcint(numbers, par, ihelp, **dd)
    overlap = OverlapLibcint(**dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        driver.setup(p, mask=mask)
        return overlap.build(driver)

    return func, pos


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")  # torch.meshgrid from batch.deflate
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, nondet_tol=1e-7)


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")  # torch.meshgrid from batch.deflate
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, nondet_tol=1e-7)
