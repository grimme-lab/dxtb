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
from tad_mctc.units import VAA2AU

from dxtb import GFN1_XTB as par
from dxtb import Calculator
from dxtb._src.components.interactions import new_efield, new_efield_grad
from dxtb._src.typing import DD, Tensor

from .samples import samples

sample_list = [
    "H",
    "H2",
    "LiH",
    "HHe",
    "H2O",
    "CH4",
    "SiH4",
    "PbH4-BiH3",
    "MB16_43_01",
]

opts = {
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "f_atol": 1.0e-9,
    "x_atol": 1.0e-9,
}

from ..conftest import DEVICE


def single(
    name: str,
    ref: Tensor,
    field_grad: Tensor,
    use_functorch: bool,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    numbers = samples[name]["numbers"].to(device)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # required for autodiff of energy w.r.t. efield
    # field_grad.requires_grad_(True)

    # create additional interaction and pass to Calculator
    efg = new_efield_grad(field_grad)
    calc = Calculator(numbers, par, interaction=[efg], opts=opts, **dd)

    qana = calc.quadrupole_analytical(
        numbers,
        positions,
        charge,
    )

    quadrupole = calc.quadrupole_numerical(
        numbers,
        positions,
        charge,
    )
    quadrupole = quadrupole.detach()
    print(ref)
    print(qana)
    print(quadrupole)

    assert pytest.approx(ref, abs=atol, rel=rtol) == quadrupole


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("use_functorch", [False, True])
def test_single(dtype: torch.dtype, name: str, use_functorch: bool) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole"].to(**dd)

    field_grad = torch.zeros((3, 3), **dd)  # * VAA2AU
    atol, rtol = 1e-3, 1e-4
    single(name, ref, field_grad, use_functorch, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LYS_xao", "C60"])
@pytest.mark.parametrize("use_functorch", [False])
def test_single_medium(
    dtype: torch.dtype, name: str, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole"].to(**dd)

    field_grad = torch.zeros((3, 3), **dd)  # * VAA2AU
    atol, rtol = 1e-2, 1e-2
    single(name, ref, field_grad, use_functorch, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("use_functorch", [False])
def test_single_field(
    dtype: torch.dtype, name: str, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole2"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    atol, rtol = 1e-3, 1e-3
    single(name, ref, field_vector, use_functorch, dd=dd, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LYS_xao", "C60"])
@pytest.mark.parametrize("use_functorch", [False])
def test_single_field_medium(
    dtype: torch.dtype, name: str, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    ref = samples[name]["quadrupole2"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    atol, rtol = 1e-2, 1e-2
    single(name, ref, field_vector, use_functorch, dd=dd, atol=atol, rtol=rtol)


def batched(
    name1: str,
    name2: str,
    refname: str,
    field_vector: Tensor,
    use_functorch: bool,
    dd: DD,
    atol: float,
    rtol: float,
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

    ref = pack(
        [
            sample1[refname].to(**dd),
            sample2[refname].to(**dd),
        ]
    )

    # required for autodiff of energy w.r.t. efield and quadrupole
    if use_functorch is True:
        field_vector.requires_grad_(True)
        pos = positions.clone().requires_grad_(True)
        field_grad = torch.zeros((3, 3), **dd, requires_grad=True)
    else:
        field_grad = None

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    efield_grad = new_efield_grad(field_grad)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    quadrupole = calc.quadrupole(
        numbers, pos, charge, use_functorch=use_functorch
    )
    quadrupole.detach_()

    assert pytest.approx(ref, abs=atol, rel=rtol) == quadrupole


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("use_functorch", [False])
def test_batch(
    dtype: torch.dtype, name1: str, name2, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU
    atol, rtol = 1e-3, 1e-3
    batched(
        name1,
        name2,
        "quadrupole",
        field_vector,
        use_functorch,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["C60"])
@pytest.mark.parametrize("use_functorch", [False])
def test_batch_medium(
    dtype: torch.dtype, name1: str, name2, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU
    atol, rtol = 1e-2, 1e-2
    batched(
        name1,
        name2,
        "quadrupole",
        field_vector,
        use_functorch,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("use_functorch", [False])
def test_batch_field(
    dtype: torch.dtype, name1: str, name2, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    atol, rtol = 1e-3, 1e-4
    batched(
        name1,
        name2,
        "quadrupole2",
        field_vector,
        use_functorch,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["C60"])
@pytest.mark.parametrize("use_functorch", [False])
def test_batch_field_medium(
    dtype: torch.dtype, name1: str, name2, use_functorch: bool
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

    field_vector = torch.tensor([-2.0, 0.5, 1.5], **dd) * VAA2AU
    atol, rtol = 1e-2, 1e-2
    batched(
        name1,
        name2,
        "quadrupole2",
        field_vector,
        use_functorch,
        dd=dd,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
def test_batch_settings(
    dtype: torch.dtype, name1: str, name2: str, scp_mode: str, mixer: str
) -> None:
    dd: DD = {"dtype": dtype, "device": device}

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
            sample1["quadrupole"].to(**dd),
            sample2["quadrupole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU

    # required for autodiff of energy w.r.t. efield and quadrupole
    field_vector.requires_grad_(True)
    pos = positions.clone().requires_grad_(True)

    efield = new_efield(field_vector)
    options = dict(opts, **{"scp_mode": scp_mode, "mixer": mixer})
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    quadrupole = calc.quadrupole(numbers, pos, charge)
    quadrupole.detach_()

    assert pytest.approx(ref, abs=1e-4) == quadrupole


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
def test_batch_unconverged(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

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
            sample1["quadrupole"].to(**dd),
            sample2["quadrupole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU

    # required for autodiff of energy w.r.t. efield and quadrupole
    field_vector.requires_grad_(True)
    pos = positions.clone().requires_grad_(True)

    # with 5 iterations, both do not converge, but pass the test
    options = dict(opts, **{"maxiter": 5, "mixer": "simple"})

    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    quadrupole = calc.quadrupole(numbers, pos, charge)
    quadrupole.detach_()

    assert pytest.approx(ref, abs=1e-2, rel=1e-3) == quadrupole


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["HHe", "LiH", "H2O"])
def test_batch_unconverged(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": device}

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
            sample1["quadrupole"].to(**dd),
            sample2["quadrupole"].to(**dd),
        ]
    )

    field_vector = torch.tensor([0.0, 0.0, 0.0], **dd) * VAA2AU

    # required for autodiff of energy w.r.t. efield and quadrupole
    field_vector.requires_grad_(True)
    pos = positions.clone().requires_grad_(True)

    # with 5 iterations, both do not converge, but pass the test
    options = dict(opts, **{"maxiter": 5, "mixer": "simple"})

    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=options, **dd)

    quadrupole = calc.quadrupole(numbers, pos, charge)
    quadrupole.detach_()

    assert pytest.approx(ref, abs=1e-2, rel=1e-3) == quadrupole
