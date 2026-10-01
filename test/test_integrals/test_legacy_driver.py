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
Regression test: ``int_driver="legacy"`` crashed in float64, because the
legacy overlap was always created in torch's default dtype (float32) and the
generalized eigensolver of the SCF refused to mix it with a float64
Hamiltonian.
"""

from __future__ import annotations

import pytest
import torch

import dxtb
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Literal

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples


@pytest.mark.skipif(
    has_libcint is False, reason="libcint interface not installed"
)
@pytest.mark.parametrize("name", ["LiH", "H2O", "SiH4"])
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_float64_energy_matches_libcint(
    gfn: Literal["gfn1", "gfn2"], name: str
) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    def energy(driver: str) -> torch.Tensor:
        par = get_param_module(gfn, **dd)
        opts = {"verbosity": 0, "int_driver": driver}
        calc = dxtb.Calculator(numbers, par, opts=opts, **dd)
        return calc.get_energy(positions)

    assert torch.allclose(
        energy("legacy"), energy("libcint"), atol=1e-11, rtol=0.0
    )


def test_float64_forces() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd).detach().clone()
    positions.requires_grad_(True)

    calc = dxtb.calculators.GFN1Calculator(
        numbers, opts={"verbosity": 0, "int_driver": "legacy"}, **dd
    )
    forces = calc.get_forces(positions)

    assert forces.dtype == torch.double
    assert torch.isfinite(forces).all()
    assert forces.abs().max() > 0
