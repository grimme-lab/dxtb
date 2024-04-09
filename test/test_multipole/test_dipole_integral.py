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

from dxtb.basis import Basis, IndexHelper
from dxtb.exlibs import libcint
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD, Tensor
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
    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    # dxtb's libcint integral
    wrapper = libcint.LibcintWrapper(atombases, ihelp)
    dxtb_dpint = libcint.int1e("j", wrapper)

    # pyscf reference integral
    assert M is not False
    mol = M(numbers, positions, parse_arg=False)
    pyscf_dpint = numpy_to_tensor(mol.intor("int1e_r_origj"), **dd)

    assert dxtb_dpint.shape == pyscf_dpint.shape
    assert pytest.approx(pyscf_dpint, abs=tol) == dxtb_dpint

    # normalize
    norm = snorm(libcint.overlap(wrapper))
    dxtb_dpint = torch.einsum("xij,i,j->xij", dxtb_dpint, norm, norm)
    pyscf_dpint = torch.einsum("xij,i,j->xij", pyscf_dpint, norm, norm)

    assert dxtb_dpint.shape == pyscf_dpint.shape
    assert pytest.approx(pyscf_dpint, abs=tol) == dxtb_dpint
