"""
Test overlap from libcint.

We cannot compare with our internal integrals or the reference integrals from
tblite, because the sorting of the p-orbitals are different.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.convert import numpy_to_tensor

from dxtb._types import DD, Tensor
from dxtb.basis import Basis, IndexHelper
from dxtb.integral.driver.libcint import impls as intor
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import is_basis_list

try:
    from dxtb.mol.external._pyscf import M
except ImportError:
    M = False

from .samples import samples

sample_list = ["H2", "LiH", "Li2", "H2O", "S", "SiH4", "MB16_43_01", "C60"]

device = None


def snorm(overlap: Tensor) -> Tensor:
    return torch.pow(overlap.diagonal(dim1=-1, dim2=-2), -0.5)


@pytest.mark.skipif(M is False, reason="PySCF not installed")
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
    assert is_basis_list(atombases)

    # dxtb's libcint integral
    wrapper = intor.LibcintWrapper(atombases, ihelp)
    dxtb_dpint = intor.int1e("j", wrapper)

    # pyscf reference integral
    assert M is not False
    mol = M(numbers, positions, parse_arg=False)
    pyscf_dpint = numpy_to_tensor(mol.intor("int1e_r_origj"), **dd)

    assert dxtb_dpint.shape == pyscf_dpint.shape
    assert pytest.approx(pyscf_dpint, abs=tol) == dxtb_dpint

    # normalize
    norm = snorm(intor.overlap(wrapper))
    dxtb_dpint = torch.einsum("xij,i,j->xij", dxtb_dpint, norm, norm)
    pyscf_dpint = torch.einsum("xij,i,j->xij", pyscf_dpint, norm, norm)

    assert dxtb_dpint.shape == pyscf_dpint.shape
    assert pytest.approx(pyscf_dpint, abs=tol) == dxtb_dpint
