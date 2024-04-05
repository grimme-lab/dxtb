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
Testing automatic energy gradient w.r.t. electric field vector.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.units import VAA2AU

from dxtb.components.interactions import new_efield
from dxtb.constants import labels
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD, Callable, Tensor
from dxtb.xtb import Calculator

from .samples import samples

opts = {
    "f_atol": 1.0e-8,
    "x_atol": 1.0e-8,
    "maxiter": 100,
    "mixer": labels.MIXER_ANDERSON,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

tol = 1e-4

# FIXME: There seem to be nultiple issues with this gradient here.
# - SiH4 fails for 0.0 (0.01 check depends on eps)
# - "ValueError: grad requires non-empty inputs." for xitorch
# - non-negligible differences between --fast and --slow
sample_list = ["H2", "H2O"]
xfields = [0.0, 1.0, -2.0]

device = None


def gradchecker(dtype: torch.dtype, name: str, xfield: float) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": device, "dtype": dtype}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * VAA2AU

    # variables to be differentiated
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        efield = new_efield(field_vector)
        calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)
        result = calc.singlepoint(numbers, positions, charge)
        energy = result.total.sum(-1)
        return energy

    return func, field_vector


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradcheck(dtype: torch.dtype, name: str, xfield: float) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradgradcheck(dtype: torch.dtype, name: str, xfield: float) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-9)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
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
    charge = torch.tensor([0.0, 0.0], **dd)

    # variables to be differentiated
    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * VAA2AU
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        efield = new_efield(field_vector)
        calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)
        result = calc.singlepoint(numbers, positions, charge)
        energy = result.total.sum(-1)
        return energy

    return func, field_vector


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, xfield)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("xfield", xfields)
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, xfield)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-8)
