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

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils import is_basis_list

if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "Li2", "S2", "H2O", "SiH4"]

# FIXME: Investigate low tolerance (normally 1e-7)!
tol = 1e-7


def num_grad(
    numbers: Tensor, ihelp: IndexHelper, positions: Tensor, intstr: str
) -> Tensor:
    # setup numerical gradient
    positions = positions.detach().clone()

    norb = int(ihelp.orbitals_per_shell.sum())
    gradient = torch.zeros(
        (3, 3, norb, norb), dtype=positions.dtype, device=positions.device
    )
    step = 1.0e-5

    def compute_integral(pos: torch.Tensor) -> torch.Tensor:
        bas = Basis(
            numbers, par, ihelp, dtype=positions.dtype, device=positions.device
        )
        atombases = bas.create_libcint(pos)
        assert is_basis_list(atombases)

        wrapper = libcint.LibcintWrapper(atombases, ihelp)
        return libcint.int1e(intstr, wrapper)

    # Loop over all atoms and their x, y, z coordinates
    for atom in range(positions.shape[0] - 1):
        print(atom, positions.shape)
        for direction in range(3):
            positions[atom, direction] += step
            ir = compute_integral(positions)

            positions[atom, direction] -= 2 * step
            il = compute_integral(positions)

            positions[atom, direction] += step
            gradient[direction] += 0.5 * (ir - il) / step

    print("")
    print("")
    print("")
    return gradient


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str):
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    positions[0] = torch.tensor([0, 0, 0], **dd)
    pos = positions.clone().requires_grad_(True)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    atombases = bas.create_libcint(pos)
    assert is_basis_list(atombases)

    INTSTR = "r0"

    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    i = libcint.int1e(INTSTR, wrapper)
    print()
    igrad = libcint.int1e(f"ip{INTSTR}", wrapper)
    igrad = igrad + igrad.mT
    print("igrad\n", igrad)
    # assert False

    print(igrad.shape)
    numgrad = num_grad(numbers, ihelp, pos, INTSTR)
    print("numgrad\n", numgrad)
    print("")
    print("")
    print("diff")
    print(igrad + numgrad)
