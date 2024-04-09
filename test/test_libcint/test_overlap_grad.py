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
Test overlap from libcint.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
import torch

from dxtb.basis import Basis, IndexHelper
from dxtb.exlibs import libcint
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD, Tensor
from dxtb.utils import is_basis_list

from ..utils import load_from_npz
from .samples import samples

ref_overlap = np.load("test/test_overlap/grad.npz")

sample_list = ["H2", "LiH", "H2O", "SiH4"]

device = None


def explicit(name: str, dd: DD, tol: float) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)  # nat
    positions = sample["positions"].to(**dd)  # nat, 3
    ref = load_from_npz(ref_overlap, name, **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    s = libcint.overlap(wrapper)
    norm = torch.pow(s.diagonal(dim1=-1, dim2=-2), -0.5)

    # (3, norb, norb)
    grad = libcint.int1e("ipovlp", wrapper)

    # normalize and move xyz dimension to last, which is required for
    # the reduction (only works with extra dimension in last)
    grad = torch.einsum("...xij,...i,...j->...ijx", grad, norm, norm)

    # (norb, norb, 3) -> (nat, norb, 3)
    grad = ihelp.reduce_orbital_to_atom(grad, dim=-3, extra=True)

    # also account for center j and negative because the integral calculates
    # the nabla w.r.t. the spatial coordinate, not the basis central position
    final_grad = -2 * grad.sum(-2)

    assert pytest.approx(ref, abs=tol) == final_grad


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_explicit(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10
    explicit(name, dd, tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_explicit_medium(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    explicit(name, dd, 1e-5)


def autograd(name: str, dd: DD, tol: float) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ref = load_from_npz(ref_overlap, name, **dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    # variable to be differentiated
    positions.requires_grad_(True)
    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    s = libcint.overlap(wrapper)
    norm = torch.pow(s.diagonal(dim1=-1, dim2=-2), -0.5)
    s = torch.einsum("...ij,...i,...j->...ij", s, norm, norm)

    (g,) = torch.autograd.grad(s.sum(), positions)
    positions.detach_()

    assert pytest.approx(ref, abs=tol) == g


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10
    autograd(name, dd, tol)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_autograd_medium(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    autograd(name, dd, 1e-5)


# Not tested, old stuff
def num_grad(bas: Basis, ihelp: IndexHelper, positions: Tensor) -> torch.Tensor:
    norb = int(ihelp.orbitals_per_shell.sum())

    # Initialize an empty tensor to store the gradients
    numerical_grad = torch.zeros(
        (norb, norb, 3), dtype=positions.dtype, device=positions.device
    )

    positions = positions.clone().detach()

    def compute_overlap(positions: torch.Tensor) -> torch.Tensor:
        atombases = bas.create_dqc(positions)
        assert is_basis_list(atombases)

        wrapper = libcint.LibcintWrapper(atombases, ihelp)
        return libcint.overlap(wrapper)

    delta = 1e-6

    # Compute the overlap integral for the original position
    s_original = compute_overlap(positions)

    # Loop over all atoms and their x, y, z coordinates
    for atom in range(positions.shape[0]):
        for direction in range(3):
            positions[..., atom, direction] += delta
            sr = compute_overlap(positions)

            positions[..., atom, direction] -= 2 * delta
            sl = compute_overlap(positions)

            positions[..., atom, direction] += delta
            numerical_grad[..., atom, direction] += (sr - sl) / delta

    # normalize and move xyz dimension to last, which is required for
    # the reduction (only works with extra dimension in last)
    norm = torch.pow(s_original.diagonal(dim1=-1, dim2=-2), -0.5)
    grad = torch.einsum("...xij,...i,...j->...ijx", numerical_grad, norm, norm)

    return grad
