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
Run tests for IR spectra.
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
from dxtb._src.typing import DD, Tensor
from dxtb.labels import INTLEVEL_DIPOLE

from ..conftest import DEVICE
from .samples import samples

slist = ["LiH"]
slist_more = ["H", "HHe", "H2O", "SiH4"]
slist_large = ["PbH4-BiH3"]
# MB16_43_01 and LYS_xao too large for testing

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
    atol: float = 1e-5,
    rtol: float = 1e-5,
    atol2: float = 1e-4,
    rtol2: float = 1e-4,
) -> None:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, field_vector, dd, atol, rtol, atol2, rtol2)


def batched(
    name1: str,
    name2: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    atol2: float = 20,
    rtol2: float = 1e-5,
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

    execute(numbers, positions, charge, field_vector, dd, atol, rtol, atol2, rtol2)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    field_vector: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
    atol2: float,
    rtol2: float,
) -> None:
    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # field is cloned and detached and updated inside
    numres = calc.raman_numerical(positions, charge)
    assert numres.freqs.grad_fn is None
    assert numres.ints.grad_fn is None
    assert numres.depol.grad_fn is None
    numfreqs = tensor_to_numpy(numres.freqs)
    numints = tensor_to_numpy(numres.ints)
    numdepol = tensor_to_numpy(numres.depol)

    # only add gradient to field_vector after numerical calculation
    field_vector.requires_grad_(True)
    calc.interactions.update_efield(field=field_vector)

    # required for autodiff of energy w.r.t. positions (Hessian)
    pos = positions.clone().detach().requires_grad_(True)

    # manual jacobian
    res1 = calc.raman(pos, charge, use_functorch=False)
    freqs1, ints1, depol1 = res1.freqs, res1.ints, res1.depol
    freqs1 = tensor_to_numpy(freqs1)
    ints1 = tensor_to_numpy(ints1)
    depol1 = tensor_to_numpy(depol1)

    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs1
    assert pytest.approx(numints, abs=atol2, rel=rtol2) == ints1
    assert pytest.approx(numdepol, abs=atol2, rel=rtol2) == depol1

    # reset (for vibration) before another AD run
    calc.reset()
    pos = positions.clone().detach().requires_grad_(True)

    # jacrev of energy
    res2 = calc.raman(pos, charge, use_functorch=True)
    freqs2, ints2, depol2 = res2.freqs, res2.ints, res2.depol
    freqs2 = tensor_to_numpy(freqs2)
    ints2 = tensor_to_numpy(ints2)
    depol2 = tensor_to_numpy(depol2)

    assert pytest.approx(numfreqs, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(freqs1, abs=atol, rel=rtol) == freqs2
    assert pytest.approx(numints, abs=atol2, rel=rtol2) == ints2
    assert pytest.approx(ints1, abs=atol2, rel=rtol2) == ints2
    assert pytest.approx(numdepol, abs=atol2, rel=rtol2) == depol2
    assert pytest.approx(depol1, abs=atol2, rel=rtol2) == depol2


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_more)
def test_single_more(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)
    single(name, field_vector, dd=dd)


# FIXME: Large deviation for all
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def skip_test_single_field(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    single(name, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def skip_test_batch_large(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU
    batched(name1, name2, field_vector, dd=dd)


# TODO: Batched derivatives are not supported yet
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def skip_test_batch_field(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    batched(name1, name2, field_vector, dd=dd)
