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

slist = ["H", "H2", "LiH", "HHe", "H2O", "CH4", "SiH4", "PbH4-BiH3"]
slist_large = ["MB16_43_01", "LYS_xao", "C60"]

opts = {
    "int_level": INTLEVEL_DIPOLE,
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1.0e-9,
    "x_atol": 1.0e-9,
}


def single(
    name: str,
    refdipole: str,  # "dipole" (no field) or "dipole2" (with field)
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    ref = samples[name][refdipole].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, ref, field_vector, dd, atol, rtol)


def batched(
    name1: str,
    name2: str,
    refdipole: str,
    field_vector: Tensor,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
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
    ref = pack(
        [
            sample1[refdipole].to(**dd),
            sample2[refdipole].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    execute(numbers, positions, charge, ref, field_vector, dd, atol, rtol)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    ref: Tensor,
    field_vector: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    npref = tensor_to_numpy(ref)

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # field is cloned and detached and updated inside
    num = tensor_to_numpy(calc.dipole_numerical(positions, charge))
    assert pytest.approx(npref, abs=atol, rel=rtol) == num

    # analytical
    dip0 = tensor_to_numpy(calc.dipole_analytical(positions, charge))
    assert pytest.approx(npref, abs=atol, rel=rtol) == dip0
    assert pytest.approx(num, abs=atol, rel=rtol) == dip0

    # required for autodiff of energy w.r.t. efield
    calc.interactions.update_efield(field=field_vector.requires_grad_(True))

    # manual jacobian
    dip1 = tensor_to_numpy(calc.dipole(positions, charge, use_functorch=False))
    assert pytest.approx(npref, abs=atol, rel=rtol) == dip1
    assert pytest.approx(num, abs=atol, rel=rtol) == dip1
    assert pytest.approx(dip0, abs=atol, rel=rtol) == dip1

    # jacrev of energy
    dip2 = tensor_to_numpy(calc.dipole(positions, charge, use_functorch=True))
    assert pytest.approx(npref, abs=atol, rel=rtol) == dip2
    assert pytest.approx(num, abs=atol, rel=rtol) == dip2
    assert pytest.approx(dip0, abs=atol, rel=rtol) == dip2
    assert pytest.approx(dip1, abs=atol, rel=rtol) == dip2


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * VAA2AU
    single(name, "dipole", field_vector, dd=dd, atol=1e-3)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * VAA2AU
    single(name, "dipole", field_vector, dd=dd, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single_field(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    single(name, "dipole2", field_vector, dd=dd, atol=1e-3, rtol=1e-3)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_field_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    single(name, "dipole2", field_vector, dd=dd, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd)  # * VAA2AU
    batched(name1, name2, "dipole", field_vector, dd=dd, atol=1e-3)


###############################################################################


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_settings(
    dtype: torch.dtype, name1: str, name2: str, scp_mode: str, mixer: str
) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

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

    ref = pack(
        [
            sample1["dipole"].to(**dd),
            sample2["dipole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU

    # required for autodiff of energy w.r.t. efield
    field_vector.requires_grad_(True)

    efield = new_efield(field_vector)
    options = dict(opts, **{"scp_mode": scp_mode, "mixer": mixer})
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    dipole = tensor_to_numpy(calc.dipole(positions, charge))
    assert pytest.approx(ref.cpu(), abs=1e-4) == dipole


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
def test_batch_unconverged(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

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

    ref = pack(
        [
            sample1["dipole"].to(**dd),
            sample2["dipole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU

    # required for autodiff of energy w.r.t. efield
    field_vector.requires_grad_(True)

    # with 5 iterations, both do not converge, but pass the test
    options = dict(opts, **{"maxiter": 5, "mixer": "simple"})

    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    dipole = tensor_to_numpy(calc.dipole(positions, charge))
    assert pytest.approx(ref.cpu(), abs=1e-2, rel=1e-3) == dipole
