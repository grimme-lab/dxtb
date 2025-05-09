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

import pytest
import torch
from tad_mctc.batch import pack
from torch.autograd import gradcheck, gradgradcheck
from torch.nn import Parameter

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, Param, ParamModule, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.factory import new_driver
from dxtb._src.integral.driver.libcint.driver import IntDriverLibcint
from dxtb._src.typing import DD, Callable, Tensor
from dxtb._src.utils import is_basis_list

if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "LiH", "H2O", "SiH4"]


def gradchecker(name: str, gfn: Param) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    """Helper function to create a gradient check function."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    par = ParamModule(gfn, **dd)

    slater = par.get("element", "H", "slater")
    assert isinstance(slater, Parameter)
    slater.requires_grad_(True)

    def func(slater: Tensor) -> Tensor:
        p = par.get("element", "H", "slater")
        assert isinstance(p, Parameter)
        p.data = slater

        ihelp = IndexHelper.from_numbers(numbers, par)
        bas = Basis(numbers, par, ihelp, **dd)

        # variable to be differentiated
        atombases = bas.create_libcint(positions)
        assert is_basis_list(atombases)

        wrapper = libcint.LibcintWrapper(atombases, ihelp)
        return libcint.overlap(wrapper)

    return func, slater


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("gfn", [GFN1_XTB, GFN2_XTB])
def test_gradcheck(name: str, gfn: Param) -> None:
    """Check single analytical against numerical gradient."""
    func, diffvars = gradchecker(name, gfn)
    assert gradcheck(func, diffvars, atol=1e-6)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("name", ["MB16_43_01"])
@pytest.mark.parametrize("gfn", [GFN1_XTB, GFN2_XTB])
def test_gradcheck_medium(name: str, gfn: Param) -> None:
    """Check single analytical against numerical gradient."""
    func, diffvars = gradchecker(name, gfn)
    assert gradcheck(func, diffvars, atol=1e-6)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("gfn", [GFN1_XTB, GFN2_XTB])
def test_gradgradcheck(name: str, gfn: Param) -> None:
    """Check single analytical against numerical gradient."""
    func, diffvars = gradchecker(name, gfn)
    assert gradgradcheck(func, diffvars, atol=1e-6)


def overlap(driver: IntDriverLibcint) -> Tensor:
    """Overlap wrapper for single and batched mode."""
    # batched mode
    if driver.ihelp.batch_mode > 0:
        assert isinstance(driver.drv, list)
        return pack([libcint.overlap(d) for d in driver.drv])

    # single mode
    assert isinstance(driver.drv, libcint.LibcintWrapper)
    return libcint.overlap(driver.drv)


def gradchecker_batch(name1: str, name2: str, gfn: Param) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    """Helper function to create a gradient check function."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    par = ParamModule(gfn, **dd)

    slater = par.get("element", "H", "slater")
    assert isinstance(slater, Parameter)
    slater.requires_grad_(True)

    def func(slater: Tensor) -> Tensor:
        p = par.get("element", "H", "slater")
        assert isinstance(p, Parameter)
        p.data = slater

        drv = new_driver(labels.INTDRIVER_LIBCINT, numbers, par)
        assert isinstance(drv, IntDriverLibcint)

        drv.setup(positions)
        return overlap(drv)

    return func, slater


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("gfn", [GFN1_XTB, GFN2_XTB])
def test_gradcheck_batch(name1: str, name2: str, gfn: Param) -> None:
    """Check batched analytical against numerical gradient."""
    func, diffvars = gradchecker_batch(name1, name2, gfn)
    assert gradcheck(func, diffvars, atol=1e-6)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("gfn", [GFN1_XTB, GFN2_XTB])
def test_gradgradcheck_batch(name1: str, name2: str, gfn: Param) -> None:
    """Check batched analytical against numerical gradient."""
    func, diffvars = gradchecker_batch(name1, name2, gfn)
    assert gradgradcheck(func, diffvars, atol=1e-6)
