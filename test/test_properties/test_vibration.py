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
Test vibrational frequencies.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy

from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD, Tensor
from dxtb.xtb import Calculator

from .samples import samples

slist = ["H", "LiH", "HHe", "H2O"]
# FIXME: Larger systems fail for modes
# slist = ["H", "LiH", "HHe", "H2O", "CH4", "SiH4", "PbH4-BiH3"]

opts = {
    "int_level": 1,
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
}

device = None


# FIXME: Autograd should also work on those
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def skip_test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # required for autodiff of energy w.r.t. efield and dipole
    positions.requires_grad_(True)

    calc = Calculator(numbers, par, opts=opts, **dd)

    def f(pos: Tensor) -> tuple[Tensor, Tensor]:
        f, m = calc.vibration(pos, charge)
        return f, m

    assert dgradcheck(f, positions)


def single(
    name: str,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, dd, atol, rtol)


def batched(
    name1: str,
    name2: str,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ],
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    execute(numbers, positions, charge, dd, atol, rtol)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    calc = Calculator(numbers, par, opts=opts, **dd)

    numfreqs, nummodes = calc.vibration_numerical(positions, charge)
    nummodes = nummodes / torch.norm(nummodes, dim=-2, keepdim=True)
    assert numfreqs.grad_fn is None
    assert nummodes.grad_fn is None

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    ###################
    # manual jacobian #
    ###################
    freqs1, modes1 = calc.vibration(pos, charge, use_functorch=False)
    modes1 = modes1 / torch.norm(modes1, dim=-2, keepdim=True)

    dot_products = torch.einsum("...ij,...ij->...j", nummodes, modes1)
    assert (torch.abs(dot_products) > 0.99).all()

    freqs1 = tensor_to_numpy(freqs1)
    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs1

    # reset before another AD run
    calc.reset()
    pos = positions.clone().detach().requires_grad_(True)

    ####################
    # jacrev of energy #
    ####################
    freqs2, modes2 = calc.vibration(pos, charge, use_functorch=True)
    modes2 = modes2 / torch.norm(modes2, dim=-2, keepdim=True)

    # check angles between modes
    dot_products = torch.einsum("...ij,...ij->...j", nummodes, modes2)
    assert (torch.abs(dot_products) > 0.99).all()

    freqs2 = tensor_to_numpy(freqs2)
    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(freqs1, abs=atol, rel=rtol) == freqs2


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    single(name, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd)
