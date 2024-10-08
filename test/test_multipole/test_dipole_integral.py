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
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.math import einsum

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint, has_pyscf
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils import is_basis_list

if has_pyscf is True:
    from dxtb._src.exlibs.pyscf.mol import M
if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from .samples import samples

slist = ["H2", "LiH", "Li2", "H2O", "S"]
slist_large = ["SiH4", "MB16_43_01", "C60"]


def snorm(overlap: Tensor) -> Tensor:
    return torch.pow(overlap.diagonal(dim1=-1, dim2=-2), -0.5)


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

    # dxtb's libcint integral
    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    dxtb_dpint = libcint.int1e("j", wrapper)

    # pyscf reference integral
    assert M is not False
    mol = M(numbers, positions, parse_arg=False)
    pyscf_dpint = numpy_to_tensor(mol.intor("int1e_r_origj"), **dd)

    assert dxtb_dpint.shape == pyscf_dpint.shape
    assert pytest.approx(pyscf_dpint.cpu(), abs=tol) == dxtb_dpint.cpu()

    # normalize
    norm = snorm(libcint.overlap(wrapper))
    dxtb_dpint = einsum("xij,i,j->xij", dxtb_dpint, norm, norm)
    pyscf_dpint = einsum("xij,i,j->xij", pyscf_dpint, norm, norm)

    assert dxtb_dpint.shape == pyscf_dpint.shape
    assert pytest.approx(pyscf_dpint.cpu(), abs=tol) == dxtb_dpint.cpu()
