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
Test Hessian.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.typing import DD, Tensor

from .samples import samples

slist = ["H", "LiH", "HHe", "H2O", "CH4", "SiH4"]
slist_large = ["PbH4-BiH3"]  # "MB16_43_01", "LYS_xao"

opts = {
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
}

device = None


# FIXME: Fails with "Numerical gradient for function expected to be zero"
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def skip_test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # required for autodiff of energy w.r.t. positions
    positions.requires_grad_(True)

    calc = Calculator(numbers, par, opts=opts, **dd)

    def f(pos: Tensor) -> Tensor:
        return calc.hessian(pos, charge)

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

    numhess = calc.hessian_numerical(positions, charge)
    assert numhess.grad_fn is None

    # jacrev of energy
    pos = positions.clone().detach().requires_grad_(True)
    hess1 = tensor_to_numpy(
        calc.hessian(
            pos,
            charge,
            use_functorch=True,
            derived_quantity="energy",
        )
    )

    # reset before another AD run
    calc.reset()

    # manual jacobian of energy
    pos = positions.clone().detach().requires_grad_(True)
    hess2 = tensor_to_numpy(
        calc.hessian(
            pos,
            charge,
            use_functorch=False,
            derived_quantity="energy",
        )
    )

    assert pytest.approx(numhess, abs=atol, rel=rtol) == hess2
    assert pytest.approx(hess1, abs=atol, rel=rtol) == hess2

    # reset before another AD run
    calc.reset()

    # jacrev of forces
    pos = positions.clone().detach().requires_grad_(True)
    hess3 = tensor_to_numpy(
        calc.hessian(
            pos,
            charge,
            use_functorch=True,
            derived_quantity="forces",
        )
    )

    assert pytest.approx(numhess, abs=atol, rel=rtol) == hess3
    assert pytest.approx(hess1, abs=atol, rel=rtol) == hess3
    assert pytest.approx(hess2, abs=atol, rel=rtol) == hess3

    # reset before another AD run
    calc.reset()

    # manual jacobian of forces
    hess4 = tensor_to_numpy(
        calc.hessian(
            pos,
            charge,
            use_functorch=False,
            derived_quantity="forces",
        )
    )

    assert pytest.approx(numhess, abs=atol, rel=rtol) == hess4
    assert pytest.approx(hess1, abs=atol, rel=rtol) == hess4
    assert pytest.approx(hess2, abs=atol, rel=rtol) == hess4
    assert pytest.approx(hess3, abs=atol, rel=rtol) == hess4


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    single(name, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    single(name, dd=dd)


# TODO: Batched Hessians are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd)


# TODO: Batched Hessians are not supported yet
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def skip_test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": device}
    batched(name1, name2, dd=dd)
