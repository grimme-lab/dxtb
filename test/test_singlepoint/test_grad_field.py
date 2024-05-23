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

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.components.interactions import new_efield
from dxtb._src.constants import labels
from dxtb._src.typing import DD, Callable, Tensor

from .samples import samples

opts = {
    "f_atol": 1.0e-8,
    "x_atol": 1.0e-8,
    "maxiter": 100,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

tol = 1e-2

# FIXME: There seem to be multiple issues with this gradient here.
# - SiH4 fails for 0.0 (0.01 check depends on eps)
# - "ValueError: grad requires non-empty inputs." for xitorch
# - non-negligible differences between --fast and --slow
sample_list = ["H2", "H2O"]
xfields = [0.0, 1.0, -2.0]

device = None


def gradchecker(dtype: torch.dtype, name: str, xfield: float, scf_mode: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": device, "dtype": dtype}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )

    # variables to be differentiated
    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * VAA2AU
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        ef = new_efield(field_vector)
        calc = Calculator(numbers, par, interaction=[ef], opts=options, **dd)
        return calc.energy(positions, charge)

    return func, field_vector


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
@pytest.mark.parametrize("scf_mode", ["implicit", "full"])
def test_gradcheck(dtype: torch.dtype, name: str, xfield: float, scf_mode: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield, scf_mode)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
@pytest.mark.parametrize("scf_mode", ["full"])
def test_gradgradcheck(
    dtype: torch.dtype, name: str, xfield: float, scf_mode: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield, scf_mode)
    assert dgradgradcheck(func, diffvars, atol=tol)


################################################################################


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float, scf_mode: str
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

    options = dict(
        opts,
        **{
            "scf_mode": scf_mode,
            "mixer": "anderson" if scf_mode == "full" else "broyden",
        },
    )

    # variables to be differentiated
    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * VAA2AU
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        ef = new_efield(field_vector)
        calc = Calculator(numbers, par, interaction=[ef], opts=options, **dd)
        return calc.energy(positions, charge)

    return func, field_vector


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("scf_mode", ["full"])
@pytest.mark.parametrize("xfield", xfields)
def test_gradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float, scf_mode: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, xfield, scf_mode)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("xfield", xfields)
@pytest.mark.parametrize("scf_mode", ["full"])
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, xfield: float, scf_mode: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, xfield, scf_mode)
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-8)
