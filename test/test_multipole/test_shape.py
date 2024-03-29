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
Testing shape of multipole integrals.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import deflate, pack
from tad_mctc.typing import DD

from dxtb.basis import Basis, IndexHelper
from dxtb.integral.driver.libcint import impls as intor
from dxtb.param import GFN1_XTB as par
from dxtb.utils import is_basis_list

from .samples import samples

sample_list = ["H2", "HHe", "LiH", "Li2", "S2", "H2O", "SiH4"]
mp_ints = ["j", "jj"]  # dipole, quadrupole

device = None


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

    wrapper = intor.LibcintWrapper(atombases, ihelp, spherical=True)
    i = intor.int1e(intstr, wrapper)

    mpdim = 3 ** len(intstr)
    assert i.shape == torch.Size((mpdim, ihelp.nao, ihelp.nao))


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
        intor.LibcintWrapper(ab, ihelp)
        for ab, ihelp in zip(atombases, ihelp)
        if is_basis_list(ab)
    ]
    i = pack([intor.int1e(intstr, wrapper) for wrapper in wrappers])

    mpdim = 3 ** len(intstr)
    assert i.shape == torch.Size((2, mpdim, _ihelp.nao, _ihelp.nao))
