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

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, IndexHelper
from dxtb._src.components.interactions.spin import factory, new_spinpolarisation
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

SINGLE_CASES = [
    pytest.param("LiH", 2, id="LiH-spin2"),
    pytest.param("SiH4", 2, id="SiH4-spin2"),
    pytest.param("MB16_43_02", 1, id="MB16_43_02-spin1"),
]


@pytest.mark.parametrize("name, spin", SINGLE_CASES)
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        (GFN1_XTB, "espgfn1"),
        (GFN2_XTB, "espgfn2"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(
    dtype: torch.dtype,
    name: str,
    spin: int,
    model_cls,
    ref_key,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    spinpol = new_spinpolarisation(numbers=numbers, **dd)
    calc = Calculator(numbers, par=model_cls, interaction=[spinpol], **dd)

    result = calc.singlepoint(
        positions, chrg=torch.tensor(0.0, **dd), spin=spin
    )
    res = result.total.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        (GFN2_XTB, "eshellgfn2"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_get_monopol_shell_energy(
    dtype: torch.dtype, name: str, model_cls, ref_key
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    qsh = torch.tensor(
        [
            [0.17175602, -0.82824398],
            [-0.33807717, -0.33807717],
            [0.16632115, -0.83367885],
        ],
        **dd,
    )

    # SpinPolarisation now expects single-channel magnetization
    qsh_mag = qsh[:, -1]

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, model_cls)
    spin = factory.new_spinpolarisation(numbers, **dd)

    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    eshell = spin.get_monopole_shell_energy(cache=cache, qsh=qsh_mag)

    at_shell = ihelp.reduce_shell_to_atom(eshell)

    assert pytest.approx(ref.cpu(), abs=tol) == at_shell.cpu()


@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        (GFN2_XTB, "potshellgfn2"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_get_monopol_shell_potential(
    dtype: torch.dtype, name: str, model_cls, ref_key
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # tblite run lih.coord --method gfn2 --spin 2 --spin-polarized --iterations 2
    # then print out the potential in spin.f90
    qsh = torch.tensor(
        [
            [0.06870241, -0.33129759],
            [-0.13523087, -0.13523087],
            [0.06652846, -0.33347154],
        ],
        **dd,
    )

    # SpinPolarisation now expects single-channel magnetization
    qsh_mag = qsh[:, -1]

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, model_cls)
    spin = factory.new_spinpolarisation(numbers, **dd)

    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    potshell = spin.get_monopole_shell_potential(cache=cache, qsh=qsh_mag)

    assert pytest.approx(ref.cpu(), abs=tol) == potshell.cpu()
