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

We cannot compare with our internal integrals or the reference integrals from
tblite, because the sorting of the p-orbitals are different.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.math import einsum

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint, has_pyscf
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils import is_basis_list

if has_libcint is True:
    from dxtb._src.exlibs import libcint
if has_pyscf is True:
    from dxtb._src.exlibs.pyscf.mol import M

from ..conftest import DEVICE
from .samples import samples

slist = ["H2", "LiH", "Li2", "H2O", "S", "SiH4"]
slist_large = ["MB16_43_01", "C60"]


def snorm(overlap: Tensor) -> Tensor:
    return torch.pow(overlap.diagonal(dim1=-1, dim2=-2), -0.5)


def extract_blocks(x: Tensor, block_sizes: list[int] | Tensor) -> list[Tensor]:
    # Initialize the start index for the first block
    start_index = 0

    # Initialize an empty list to store the blocks
    blocks: list[Tensor] = []

    if isinstance(block_sizes, Tensor):
        assert block_sizes.ndim == 1
        block_sizes = block_sizes.tolist()

    # Iterate over each block
    for block_size in block_sizes:
        # Generate the indices for the elements in the current block
        indices = start_index + torch.arange(block_size)

        # Extract the block and append it to the list
        blocks.append(x[indices, :][:, indices])

        # Update the start index for the next block
        start_index += block_size

    return blocks


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    run_single(dtype, name)


@pytest.mark.large
@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_large(dtype: torch.dtype, name: str) -> None:
    run_single(dtype, name)


def run_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_libcint(positions)
    assert is_basis_list(atombases)

    # dxtb's libcint overlap
    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    dxtb_overlap = libcint.overlap(wrapper)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions, parse_arg=False)  # type: ignore
    pyscf_overlap = numpy_to_tensor(mol.intor("int1e_ovlp"), **dd)

    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap.cpu(), abs=tol) == dxtb_overlap.cpu()

    # normalize dxtb's overlap
    norm = snorm(dxtb_overlap)
    dxtb_overlap = einsum("ij,i,j->ij", dxtb_overlap, norm, norm)

    # normalize PySCF's reference overlap
    norm = snorm(pyscf_overlap)
    pyscf_overlap = einsum("ij,i,j->ij", pyscf_overlap, norm, norm)
    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap.cpu(), abs=tol) == dxtb_overlap.cpu()


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Batched overlap using non-batched setup, i.e., one huge matrix is
    calculated that is only populated on the diagonal.
    """
    run_batch(dtype, name1, name2)


@pytest.mark.large
@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def test_large_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Batched overlap using non-batched setup, i.e., one huge matrix is
    calculated that is only populated on the diagonal.
    """
    run_batch(dtype, name1, name2)


def run_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = torch.cat(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = torch.cat(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd) + 1000,  # move!
        ),
        dim=0,
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_libcint(positions)
    assert is_basis_list(atombases)

    # dxtb's libcint overlap
    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    dxtb_overlap = libcint.overlap(wrapper)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions, parse_arg=False)  # type: ignore
    pyscf_overlap = numpy_to_tensor(mol.intor("int1e_ovlp"), **dd)

    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap.cpu(), abs=tol) == dxtb_overlap.cpu()

    # normalize dxtb's overlap
    norm = snorm(dxtb_overlap)
    dxtb_overlap = einsum("ij,i,j->ij", dxtb_overlap, norm, norm)

    # normalize PySCF's reference overlap
    norm = snorm(pyscf_overlap)
    pyscf_overlap = einsum("ij,i,j->ij", pyscf_overlap, norm, norm)

    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap.cpu(), abs=tol) == dxtb_overlap.cpu()

    # we could also extract the blocks and pack them as usual
    n = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    ihelp2 = IndexHelper.from_numbers(n, par)
    sizes = ihelp2.orbitals_per_shell.sum(-1)
    out = extract_blocks(dxtb_overlap, sizes)
    s_packed = pack(out)

    max_size = int(ihelp2.orbitals_per_shell.sum(-1).max())
    assert s_packed.shape == torch.Size((2, max_size, max_size))


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_grad(dtype: torch.dtype, name: str) -> None:
    run_grad(dtype, name)


@pytest.mark.large
@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_large_grad(dtype: torch.dtype, name: str) -> None:
    run_grad(dtype, name)


def run_grad(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_libcint(positions)
    assert is_basis_list(atombases)

    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    int1 = libcint.int1e("ipovlp", wrapper)

    mol = M(numbers, positions, parse_arg=False)  # type: ignore
    int2 = numpy_to_tensor(mol.intor("int1e_ipovlp"), **dd)

    assert int1.shape == int2.shape
    assert pytest.approx(int2.cpu(), abs=tol) == int1.cpu()
