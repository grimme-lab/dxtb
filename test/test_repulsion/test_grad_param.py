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
Run tests for repulsion parameter gradient.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import Repulsion
from dxtb._src.param import get_elem_param
from dxtb._src.typing import DD, Callable, Tensor
from dxtb._src.utils import batch

from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]

tol = 1e-8

device = None


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    # variables to be differentiated
    _arep = get_elem_param(
        torch.unique(numbers),
        par.element,
        "arep",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _zeff = get_elem_param(
        torch.unique(numbers),
        par.element,
        "zeff",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _kexp = torch.tensor(par.repulsion.effective.kexp, **dd, requires_grad=True)

    def func(arep: Tensor, zeff: Tensor, kexp: Tensor) -> Tensor:
        rep = Repulsion(arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)
        return rep.get_energy(positions, cache)

    return func, (_arep, _zeff, _kexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(dtype: torch.dtype, name1: str, name2: str) -> tuple[
    Callable[[Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor],  # differentiable variables
]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, par)

    # variables to be differentiated
    _arep = get_elem_param(
        torch.unique(numbers),
        par.element,
        "arep",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _zeff = get_elem_param(
        torch.unique(numbers),
        par.element,
        "zeff",
        pad_val=0,
        **dd,
        requires_grad=True,
    )
    _kexp = torch.tensor(par.repulsion.effective.kexp, **dd, requires_grad=True)

    def func(arep: Tensor, zeff: Tensor, kexp: Tensor) -> Tensor:
        rep = Repulsion(arep, zeff, kexp, **dd)
        cache = rep.get_cache(numbers, ihelp)
        return rep.get_energy(positions, cache)

    return func, (_arep, _zeff, _kexp)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert gradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)

    # Although, we add an epsilon within the square root of arep to avoid
    # taking the square root at zero, the step size in gradgradcheck is larger
    # than this epsilon, which leads to nan's again. As this does not effect
    # regular second order derivatives, we omit the arep parameter here.
    diffvars[0].requires_grad_(False)

    assert gradgradcheck(func, diffvars, atol=tol)
