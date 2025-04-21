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
Test coefficient derivative with libcint.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.autograd import dgradcheck

from dxtb import GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils import is_basis_list

if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "LiH", "H2O", "SiH4"]


def autograd(name: str, dd: DD, tol: float) -> None:
    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    par = ParamModule(GFN2_XTB, **dd)

    slater = par.get("element", "H", "slater")
    slater.requires_grad_(True)

    def func(slater: Tensor) -> Tensor:
        par.get("element", "H", "slater").data = slater

        ihelp = IndexHelper.from_numbers(numbers, par)
        bas = Basis(numbers, par, ihelp, **dd)

        # variable to be differentiated
        atombases = bas.create_libcint(positions)
        assert is_basis_list(atombases)

        wrapper = libcint.LibcintWrapper(atombases, ihelp)
        return libcint.overlap(wrapper)

    assert dgradcheck(func, slater, atol=tol)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps) * 10
    autograd(name, dd, tol)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_autograd_medium(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    autograd(name, dd, 1e-5)
