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
Testing automatic energy gradient w.r.t. electric field gradient.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.components.interactions import new_efield, new_efield_grad
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from .samples import samples

opts = {
    "int_level": labels.INTLEVEL_DIPOLE,
    "f_atol": 1.0e-8,
    "x_atol": 1.0e-8,
    "maxiter": 100,
    "mixer": labels.MIXER_ANDERSON,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

tol = 1e-4

sample_list = ["H2", "H2O", "SiH4"]


def gradchecker(dtype: torch.dtype, name: str, gfn: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * units.VAA2AU
    field_grad = torch.zeros((3, 3), **dd)

    # variables to be differentiated
    field_grad.requires_grad_(True)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        raise ValueError(f"Unknown GFN: {gfn}")

    def func(fieldgrad: Tensor) -> Tensor:
        efield = new_efield(field_vector)
        efield_grad = new_efield_grad(fieldgrad)
        calc = Calculator(
            numbers, par, interaction=[efield, efield_grad], opts=opts, **dd
        )
        result = calc.singlepoint(positions, charge)
        energy = result.total.sum(-1)
        return energy

    return func, field_grad


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_gradcheck(dtype: torch.dtype, name: str, gfn: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, gfn)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_gradgradcheck(dtype: torch.dtype, name: str, gfn: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, gfn)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-9)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, gfn: str
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * units.VAA2AU
    field_grad = torch.zeros((3, 3), **dd)

    # variables to be differentiated
    field_grad.requires_grad_(True)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        raise ValueError(f"Unknown GFN: {gfn}")

    def func(fieldgrad: Tensor) -> Tensor:
        efield = new_efield(field_vector)
        efield_grad = new_efield_grad(fieldgrad)
        calc = Calculator(
            numbers, par, interaction=[efield, efield_grad], opts=opts, **dd
        )
        result = calc.singlepoint(positions, charge)
        energy = result.total.sum(-1)
        return energy

    return func, field_grad


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_gradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, gfn: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, gfn)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, gfn: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, gfn)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-8)
