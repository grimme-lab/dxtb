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
from tad_mctc.typing import DD, Tensor

from dxtb.basis import Basis, IndexHelper
from dxtb.integral.driver.libcint import impls as intor
from dxtb.param import GFN1_XTB as par
from dxtb.utils import is_basis_list

try:
    from dxtb.mol.external._pyscf import M

    pyscf = True
except ImportError:
    pyscf = False

from .samples import samples

sample_list = ["H2", "LiH", "Li2", "H2O", "S", "SiH4", "MB16_43_01", "C60"]

device = None


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


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    # dxtb's libcint overlap
    wrapper = intor.LibcintWrapper(atombases, ihelp)
    dxtb_overlap = intor.overlap(wrapper)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions, parse_arg=False)  # type: ignore
    pyscf_overlap = numpy_to_tensor(mol.intor("int1e_ovlp"), **dd)

    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap, abs=tol) == dxtb_overlap

    # normalize dxtb's overlap
    norm = snorm(dxtb_overlap)
    dxtb_overlap = torch.einsum("ij,i,j->ij", dxtb_overlap, norm, norm)

    # normalize PySCF's reference overlap
    norm = snorm(pyscf_overlap)
    pyscf_overlap = torch.einsum("ij,i,j->ij", pyscf_overlap, norm, norm)
    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap, abs=tol) == dxtb_overlap


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Batched overlap using non-batched setup, i.e., one huge matrix is
    calculated that is only populated on the diagonal.
    """
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = torch.cat(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
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
    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    # dxtb's libcint overlap
    wrapper = intor.LibcintWrapper(atombases, ihelp)
    dxtb_overlap = intor.overlap(wrapper)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions, parse_arg=False)  # type: ignore
    pyscf_overlap = numpy_to_tensor(mol.intor("int1e_ovlp"), **dd)

    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap, abs=tol) == dxtb_overlap

    # normalize dxtb's overlap
    norm = snorm(dxtb_overlap)
    dxtb_overlap = torch.einsum("ij,i,j->ij", dxtb_overlap, norm, norm)

    # normalize PySCF's reference overlap
    norm = snorm(pyscf_overlap)
    pyscf_overlap = torch.einsum("ij,i,j->ij", pyscf_overlap, norm, norm)

    assert dxtb_overlap.shape == pyscf_overlap.shape
    assert pytest.approx(pyscf_overlap, abs=tol) == dxtb_overlap

    # we could also extract the blocks and pack them as usual
    n = pack((sample1["numbers"].to(device), sample2["numbers"].to(device)))
    ihelp2 = IndexHelper.from_numbers(n, par)
    sizes = ihelp2.orbitals_per_shell.sum(-1)
    out = extract_blocks(dxtb_overlap, sizes)
    s_packed = pack(out)

    max_size = int(ihelp2.orbitals_per_shell.sum(-1).max())
    assert s_packed.shape == torch.Size((2, max_size, max_size))


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    wrapper = intor.LibcintWrapper(atombases, ihelp)
    int1 = intor.int1e("ipovlp", wrapper)

    mol = M(numbers, positions, parse_arg=False)  # type: ignore
    int2 = numpy_to_tensor(mol.intor("int1e_ipovlp"), **dd)

    assert int1.shape == int2.shape
    assert pytest.approx(int2, abs=tol) == int1
