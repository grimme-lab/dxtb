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
Run tests for energy contribution from instantaneous electric field.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.units import VAA2AU

from dxtb.components.interactions import new_efield
from dxtb.constants import labels
from dxtb.param import GFN1_XTB
from dxtb.typing import DD
from dxtb.xtb import Calculator

from .samples import samples

sample_list = ["MB16_43_01"]
sample_list = ["LiH", "SiH4"]

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
}

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    ref = sample["energy"].to(**dd)
    # ref1 = sample["energy_monopole"].to(**dd)

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * VAA2AU
    efield = new_efield(field_vector)
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=opts, **dd)

    result = calc.singlepoint(positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize(
    "scf_mode", [labels.SCF_MODE_IMPLICIT_NON_PURE, labels.SCF_MODE_FULL]
)
def test_batch(dtype: torch.dtype, name1: str, name2: str, scf_mode: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0], **dd)

    ref1 = torch.stack(
        [
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
        ],
    )

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * VAA2AU
    efield = new_efield(field_vector)
    options = dict(opts, **{"scf_mode": scf_mode, "mixer": "anderson"})
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    assert pytest.approx(ref1, abs=tol, rel=tol) == result.total.sum(-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("name3", sample_list)
@pytest.mark.parametrize(
    "scf_mode", [labels.SCF_MODE_IMPLICIT_NON_PURE, labels.SCF_MODE_FULL]
)
def test_batch_three(
    dtype: torch.dtype, name1: str, name2: str, name3: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2, sample3 = samples[name1], samples[name2], samples[name3]
    numbers = pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
            sample3["numbers"].to(device),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
            sample3["positions"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    ref = torch.stack(
        [
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
            sample3["energy"].to(**dd),
        ],
    )

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * VAA2AU
    efield = new_efield(field_vector)
    options = dict(opts, **{"scf_mode": scf_mode, "mixer": "anderson"})
    calc = Calculator(numbers, GFN1_XTB, interaction=[efield], opts=options, **dd)

    result = calc.singlepoint(positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.total.sum(-1)
