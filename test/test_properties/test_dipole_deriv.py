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
Run tests for geometric polarizability derivative.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.units import VAA2AU

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.components.interactions import new_efield
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb.labels import INTLEVEL_DIPOLE

from ..conftest import DEVICE
from .samples import samples

slist = ["LiH"]
slist_more = ["H", "HHe", "H2O", "CH4", "PbH4-BiH3", "MB16_43_01"]
slist_large = ["PbH4-BiH3", "MB16_43_01"]

opts = {
    "int_level": INTLEVEL_DIPOLE,
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
}


def single(
    name: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, field_vector, dd, atol, rtol)


def batched(
    name1: str,
    name2: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ],
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    execute(numbers, positions, charge, field_vector, dd, atol, rtol)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    field_vector: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # field is cloned and detached and updated inside
    num = calc.dipole_deriv_numerical(positions, charge)

    # required for autodiff of energy w.r.t. efield
    calc.interactions.update_efield(field=field_vector.requires_grad_(True))

    # manual jacobian with analytical dipole derivative
    dipder1 = tensor_to_numpy(
        calc.dipole_deriv(
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=True,
            use_functorch=False,
        )
    )
    assert pytest.approx(num.cpu(), abs=atol, rel=rtol) == dipder1

    # applying AD twice requires detaching
    calc.reset()

    # manual jacobian with AD dipole moment
    dipder2 = tensor_to_numpy(
        calc.dipole_deriv(
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=False,
            use_functorch=False,
        )
    )
    assert pytest.approx(num.cpu(), abs=atol, rel=rtol) == dipder2

    # applying AD twice requires detaching
    calc.reset()

    # jacrev of analytical dipole moment
    dipder3 = tensor_to_numpy(
        calc.dipole_deriv(
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=True,
            use_functorch=True,
        )
    )
    assert pytest.approx(num.cpu(), abs=atol, rel=rtol) == dipder3

    # applying AD twice requires detaching
    calc.reset()

    # jacrev of AD dipole moment
    dipder4 = tensor_to_numpy(
        calc.dipole_deriv(
            positions.detach().clone().requires_grad_(True),
            charge,
            use_analytical=False,
            use_functorch=True,
        )
    )
    assert pytest.approx(num.cpu(), abs=atol, rel=rtol) == dipder4

    assert pytest.approx(dipder1, abs=atol) == dipder2
    assert pytest.approx(dipder1, abs=atol) == dipder3
    assert pytest.approx(dipder1, abs=atol) == dipder4


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.large
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_more)
def test_single_more(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.large
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single_field(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    single(name, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.large
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def skip_test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch_field(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    batched(name1, name2, field_vector, dd=dd)
