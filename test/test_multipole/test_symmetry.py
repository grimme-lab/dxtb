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
Testing symmetry of (higher) multipole integrals.
"""

from __future__ import annotations

from itertools import permutations

import pytest
import torch
from tad_mctc.batch import deflate, pack

from dxtb.basis import Basis, IndexHelper
from dxtb.exlibs import libcint
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD, Tensor
from dxtb.utils import is_basis_list

from .samples import samples

sample_list = ["H2", "HHe", "LiH", "S2", "H2O", "SiH4"]
mp_ints = ["jj"]

device = None


def check_multipole_symmetry(multipole_tensor: Tensor) -> bool:
    """
    Check the symmetry of a multipole moment tensor.

    Args:
    multipole_tensor: A PyTorch tensor that represents a multipole moment.

    Returns:
    A Boolean value indicating whether the tensor is symmetric.
    """

    # Get the number of dimensions in the tensor
    ndim = len(multipole_tensor.shape)

    # Get all permutations of the first (ndim-2) dimensions
    perms = permutations(range(ndim - 2))

    # Check symmetry for all permutations
    is_symmetric = all(
        torch.allclose(
            multipole_tensor, multipole_tensor.permute(*perm, ndim - 2, ndim - 1)
        )
        for perm in perms
    )

    return is_symmetric


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("intstr", mp_ints)
def test_single(dtype: torch.dtype, intstr: str, name: str) -> None:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    wrapper = libcint.LibcintWrapper(atombases, ihelp, spherical=True)
    i = libcint.int1e(intstr, wrapper)

    nao = wrapper.nao()
    mpdims = len(intstr) * (3,)
    i = i.reshape((*mpdims, nao, nao))

    assert i.shape == torch.Size((*mpdims, ihelp.nao, ihelp.nao))
    assert check_multipole_symmetry(i) is True


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("intstr", mp_ints)
def test_batch(dtype: torch.dtype, name1: str, name2: str, intstr: str) -> None:
    """Prepare gradient check from `torch.autograd`."""
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

    _ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, _ihelp, **dd)
    atombases = bas.create_dqc(positions)

    # batched IndexHelper does not yet work with LibcintWrapper
    ihelp = [IndexHelper.from_numbers(deflate(number), par) for number in numbers]

    wrappers = [
        libcint.LibcintWrapper(ab, ihelp)
        for ab, ihelp in zip(atombases, ihelp)
        if is_basis_list(ab)
    ]

    mpdims = len(intstr) * (3,)

    int_list = []
    for wrapper in wrappers:
        i = libcint.int1e(intstr, wrapper)

        nao = wrapper.nao()
        i = i.reshape((*mpdims, nao, nao))

        int_list.append(i)

    i = pack(int_list)

    assert i.shape == torch.Size((2, *mpdims, _ihelp.nao, _ihelp.nao))
