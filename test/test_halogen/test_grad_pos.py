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
Testing halogen bond correction gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import new_halogen
from dxtb._src.typing import DD, Callable, Tensor

from .samples import samples

# "LYS_xao" must be the last one as we have to manually exclude it for the
# `backward` and `gradgradcheck` check because the gradient is zero
sample_list = ["br2nh3", "br2och2", "LYS_xao"]

tol = 1e-8

device = None


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    def func(pos: Tensor) -> Tensor:
        return xb.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list[:-1])
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(dtype: torch.dtype, name1: str, name2: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    def func(pos: Tensor) -> Tensor:
        return xb.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["br2nh3", "finch", "LYS_xao"])
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["gradient"].to(**dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    grad_autograd = xb.get_gradient(energy, positions)

    positions.detach_()
    grad_autograd.detach_()

    assert pytest.approx(ref, abs=tol * 10) == grad_autograd


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_autograd_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["gradient"].to(**dd),
            sample2["gradient"].to(**dd),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    grad_autograd = xb.get_gradient(energy, positions)

    positions.detach_()
    grad_autograd.detach_()

    assert pytest.approx(ref, abs=tol * 10) == grad_autograd


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list[:-1])
def test_backward(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = sample["gradient"].to(**dd)

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    energy.sum().backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref, abs=tol * 10) == grad_backward


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["br2nh3"])
@pytest.mark.parametrize("name2", sample_list)
def test_backward_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["gradient"].to(**dd),
            sample2["gradient"].to(**dd),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    xb = new_halogen(numbers, par, **dd)
    assert xb is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = xb.get_cache(numbers, ihelp)

    energy = xb.get_energy(positions, cache)
    energy.sum().backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    assert pytest.approx(ref, abs=tol * 10) == grad_backward

    positions.detach_()
