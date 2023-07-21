"""
Test overlap from libcint.

We cannot compare with our internal integrals or the reference integrals from
tblite, because the sorting of the p-orbitals are different.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.basis import Basis, IndexHelper
from dxtb.integral.libcint import LibcintWrapper, intor
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch, numpy_to_tensor

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


@pytest.mark.skipif(pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)

    print(atombases)

    # dxtb's libcint overlap
    wrapper = LibcintWrapper(atombases, ihelp)
    dxtb_overlap = intor.overlap(wrapper)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions, parse_arg=False)
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
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers1 = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    print(numbers1)
    numbers = torch.cat(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = torch.cat(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd) + 1000,
        ),
        dim=0,
    )

    print(numbers)
    print(positions)
    ihelp1 = IndexHelper.from_numbers(numbers1, get_elem_angular(par.element))
    bas = Basis(numbers1, par, ihelp1, **dd)
    print(bas.numbers)
    print(bas.ngauss)
    print("")
    print("")
    print(ihelp1.orbitals_per_shell.sum(-1))

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)
    print(ihelp.orbitals_per_shell)
    print(ihelp.orbitals_per_shell.sum(-1))

    # dxtb's libcint overlap
    wrapper = LibcintWrapper(atombases, ihelp)
    dxtb_overlap = intor.overlap(wrapper)

    print(dxtb_overlap.shape)
    print(dxtb_overlap)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions, parse_arg=False)
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
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)

    wrapper = LibcintWrapper(atombases, ihelp)
    int1 = intor.int1e("ipovlp", wrapper)

    mol = M(numbers, positions, parse_arg=False)
    int2 = numpy_to_tensor(mol.intor("int1e_ipovlp"), **dd)

    assert int1.shape == int2.shape
    assert pytest.approx(int2, abs=tol) == int1
