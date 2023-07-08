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
from dxtb.mol.external._pyscf import M
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import numpy_to_tensor

from .samples import samples

sample_list = ["H2", "LiH", "Li2", "H2O", "S", "SiH4", "MB16_43_01", "C60"]

device = None


def snorm(overlap: Tensor) -> Tensor:
    return torch.pow(overlap.diagonal(dim1=-1, dim2=-2), -0.5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp.angular, **dd)
    atombases = bas.create_dqc(positions, ihelp)

    # dxtb's libcint overlap
    wrapper = LibcintWrapper(atombases, ihelp)
    dxtb_overlap = intor.overlap(wrapper)

    # pyscf reference overlap ("sph" needed, implicit in dxtb)
    mol = M(numbers, positions)
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


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 1e-2

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp.angular, **dd)
    atombases = bas.create_dqc(positions, ihelp)

    wrapper = LibcintWrapper(atombases, ihelp)
    int1 = intor.int1e("ipovlp", wrapper)

    mol = M(numbers, positions)
    int2 = numpy_to_tensor(mol.intor("int1e_ipovlp"), **dd)

    assert int1.shape == int2.shape
    assert pytest.approx(int2, abs=tol) == int1
