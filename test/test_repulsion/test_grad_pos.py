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
Run tests for nuclear repulsion gradient.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.components.classicals import Repulsion, new_repulsion
from dxtb._src.components.classicals.repulsion.base import BaseRepulsionCache
from dxtb._src.components.classicals.repulsion.rep import (
    repulsion_energy,
    repulsion_gradient,
)
from dxtb._src.typing import DD, Callable, Tensor
from tad_mctc.batch import pack

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]

tol = 1e-7


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_backward_vs_tblite(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["gfn1_grad"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["H2O", "SiH4"])
def test_backward_batch_vs_tblite(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
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
            sample1["gfn1_grad"].to(**dd),
            sample2["gfn1_grad"].to(**dd),
        ]
    )

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache))
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_backward_vs_analytical(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    e = repulsion_energy(positions, cache.mask, cache.arep, cache.kexp, cache.zeff)
    grad_analytical = repulsion_gradient(
        e, positions, cache.mask, cache.arep, cache.kexp, reduced=True
    )

    # automatic gradient
    positions.requires_grad_(True)
    energy = torch.sum(rep.get_energy(positions, cache), dim=-1)
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(grad_analytical.cpu(), abs=tol) == grad_backward.cpu()


def calc_numerical_gradient(
    positions: Tensor, rep: Repulsion, cache: BaseRepulsionCache
) -> Tensor:
    """Calculate gradient numerically for reference."""

    n_atoms = positions.shape[0]

    # setup numerical gradient
    gradient = torch.zeros((n_atoms, 3), dtype=positions.dtype)
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            er, el = 0.0, 0.0

            positions[i, j] += step
            er = rep.get_energy(positions, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            el = rep.get_energy(positions, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_analytical_vs_numerical(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    atol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    # analytical gradient
    e = repulsion_energy(positions, cache.mask, cache.arep, cache.kexp, cache.zeff)
    grad_analytical = repulsion_gradient(
        e, positions, cache.mask, cache.arep, cache.kexp, reduced=True
    )

    # numerical gradient
    grad_num = calc_numerical_gradient(positions, rep, cache)
    assert pytest.approx(grad_num.cpu(), abs=atol) == grad_analytical.cpu()


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    assert par.repulsion is not None

    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None
    cache = rep.get_cache(numbers, ihelp)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return rep.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    ihelp = IndexHelper.from_numbers(numbers, par)

    rep = new_repulsion(numbers, par, **dd)
    assert rep is not None
    cache = rep.get_cache(numbers, ihelp)

    # variables to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return rep.get_energy(pos, cache)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list + ["MB16_43_03"])
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list + ["MB16_43_03"])
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)
