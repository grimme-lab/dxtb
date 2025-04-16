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
# pylint: disable=protected-access
"""
Run tests for gradient from isotropic second-order electrostatic energy (ES2).
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.components.interactions.coulomb import secondorder as es2
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["MB16_43_07", "MB16_43_08", "SiH4", "LiH"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test gradient for some samples from MB16_43."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charges = sample["q"].to(**dd)
    ref = sample["grad"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    es = es2.new_es2(torch.unique(numbers), GFN1_XTB, shell_resolved=True, **dd)
    assert es is not None

    es.cache_disable()
    cache = es.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    # atom gradient should be zero
    grad_atom = es._get_atom_gradient(numbers, positions, charges, cache)
    assert (torch.zeros_like(positions) == grad_atom).all()

    # analytical (old)
    grad = es._get_shell_gradient(numbers, positions, charges, cache, ihelp)
    assert pytest.approx(grad.cpu(), abs=tol) == ref.cpu()

    # numerical
    num_grad = calc_numerical_gradient(numbers, positions, ihelp, charges)
    assert pytest.approx(ref.cpu(), abs=tol) == num_grad.cpu()
    assert pytest.approx(num_grad.cpu(), abs=tol) == grad.cpu()

    # automatic
    pos = positions.clone().requires_grad_(True)
    mat = es.get_shell_coulomb_matrix(numbers, pos, ihelp)
    energy = 0.5 * mat * charges.unsqueeze(-1) * charges.unsqueeze(-2)
    (agrad,) = torch.autograd.grad(energy.sum(), pos)
    assert pytest.approx(ref.cpu(), abs=tol) == agrad.cpu()

    # analytical (automatic)
    cache = es.get_cache(numbers=numbers, positions=pos, ihelp=ihelp)  # recalc
    egrad = es.get_shell_gradient(charges, pos, cache)
    egrad.detach_()
    assert pytest.approx(ref.cpu(), abs=tol) == egrad.cpu()
    assert pytest.approx(egrad.cpu(), abs=tol) == agrad.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Test gradient for multiple systems."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    charges = pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["grad"].to(**dd),
            sample2["grad"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    es = es2.new_es2(torch.unique(numbers), GFN1_XTB, shell_resolved=True, **dd)
    assert es is not None

    es.cache_disable()
    cache = es.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    # analytical (old)
    grad = es._get_shell_gradient(numbers, positions, charges, cache, ihelp)
    assert pytest.approx(grad.cpu(), abs=tol) == ref.cpu()

    # automatic
    pos = positions.clone().requires_grad_(True)
    mat = es.get_shell_coulomb_matrix(numbers, pos, ihelp)
    energy = 0.5 * mat * charges.unsqueeze(-1) * charges.unsqueeze(-2)
    (agrad,) = torch.autograd.grad(energy.sum(), pos)
    assert pytest.approx(ref.cpu(), abs=tol) == agrad.cpu()

    # analytical (automatic)
    cache = es.get_cache(numbers=numbers, positions=pos, ihelp=ihelp)  # recalc
    egrad = es.get_shell_gradient(charges, pos, cache)
    egrad.detach_()
    assert pytest.approx(ref.cpu(), abs=tol) == egrad.cpu()
    assert pytest.approx(egrad.cpu(), abs=tol) == agrad.cpu()


def calc_numerical_gradient(
    numbers: Tensor, positions: Tensor, ihelp: IndexHelper, charges: Tensor
) -> Tensor:
    """Calculate gradient numerically for reference."""
    dtype = torch.double
    es = es2.new_es2(
        torch.unique(numbers),
        GFN1_XTB,
        shell_resolved=True,
        dtype=dtype,
        device=positions.device,
    )
    assert es is not None

    positions = positions.type(dtype)
    charges = charges.type(dtype)

    # setup numerical gradient
    gradient = torch.zeros_like(positions)
    step = 1.0e-6

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            cache = es.get_cache(
                numbers=numbers, positions=positions, ihelp=ihelp
            )
            er = es.get_monopole_shell_energy(cache, charges)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            cache = es.get_cache(
                numbers=numbers, positions=positions, ihelp=ihelp
            )
            el = es.get_monopole_shell_energy(cache, charges)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
