# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Run spin-polarized gradient tests against tblite references stored in
``test/test_spinpol/samples.py``.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.components.interactions.spin import new_spinpolarisation
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

CASES = [
    pytest.param("LiH", 2, id="LiH-spin2"),
    pytest.param("SiH4", 2, id="SiH4-spin2"),
]

METHODS = [
    pytest.param(GFN1_XTB, "gspgfn1", id="gfn1"),
    pytest.param(
        GFN2_XTB,
        "gspgfn2",
        id="gfn2",
        marks=pytest.mark.skipif(
            not has_libcint, reason="libcint not available"
        ),
    ),
]


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name, spin", CASES)
@pytest.mark.parametrize("model_cls, ref_key", METHODS)
def test_backward_against_tblite(
    name: str, spin: int, model_cls, ref_key: str
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).clone().requires_grad_(True)
    ref = sample[ref_key].to(**dd)

    spinpol = new_spinpolarisation(numbers=numbers, **dd)
    calc = Calculator(
        numbers,
        par=model_cls,
        interaction=[spinpol],
        opts={"verbosity": 0},
        **dd,
    )

    result = calc.singlepoint(
        positions, chrg=torch.tensor(0.0, **dd), spin=spin
    )
    result.total.sum(-1).backward()

    assert positions.grad is not None
    autograd = positions.grad.clone()
    assert pytest.approx(ref.cpu(), abs=1e-5, rel=2e-4) == autograd.cpu()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name, spin", CASES)
@pytest.mark.parametrize("model_cls, ref_key", METHODS)
def test_forces_against_tblite(
    name: str, spin: int, model_cls, ref_key: str
) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).clone().requires_grad_(True)
    ref = sample[ref_key].to(**dd)

    spinpol = new_spinpolarisation(numbers=numbers, **dd)
    calc = Calculator(
        numbers,
        par=model_cls,
        interaction=[spinpol],
        opts={"verbosity": 0},
        **dd,
    )

    forces = calc.forces(positions, chrg=torch.tensor(0.0, **dd), spin=spin)
    assert pytest.approx((-ref).cpu(), abs=1e-5, rel=2e-4) == forces.cpu()
