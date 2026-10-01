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
Test properties (multipole moments, bond orders) of spin-polarized
calculations.

Reference values obtained with tblite 0.7.0 (see `samples.py`).
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.components.interactions.spin import new_spin_polarization
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

opts = {"verbosity": 0}
store = {"store_overlap": True, "store_density": True}


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_multipoles(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1e-4 if dtype == torch.double else 1e-3

    sample = samples["OH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    spin = sample["spin"].to(**dd)

    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(numbers, GFN2_XTB, interaction=[spinpol], opts=opts, **dd)
    result = calc.singlepoint(positions, spin=spin)

    # atomic multipoles in charge/magnetization representation
    assert result.charges.dipole is not None
    ref = sample["dpgfn2"].to(**dd)
    assert pytest.approx(ref.cpu(), abs=tol) == result.charges.dipole.cpu()

    assert result.charges.quad is not None
    ref = sample["qpgfn2"].to(**dd)
    assert pytest.approx(ref.cpu(), abs=tol) == result.charges.quad.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bond_orders(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1e-5 if dtype == torch.double else 1e-4

    sample = samples["OH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    spin = sample["spin"].to(**dd)
    ref = sample["wbogfn1"].to(**dd)

    spinpol = new_spin_polarization(numbers, **dd)
    calc = Calculator(numbers, GFN1_XTB, interaction=[spinpol], opts=opts, **dd)
    wbo = calc.bond_orders(positions, spin=spin, **store)

    # bond orders in charge/magnetization representation
    assert wbo.shape == (2, 2, 2)
    assert pytest.approx(ref.cpu(), abs=tol) == wbo.cpu()


def test_bond_orders_shape() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample = samples["OH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    chrg = torch.tensor(-1.0, **dd)  # closed-shell hydroxide

    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    restricted = calc.bond_orders(positions, chrg, **store)
    assert restricted.shape == (2, 2)

    calc = Calculator(numbers, GFN1_XTB, opts={**opts, "uhf_mode": True}, **dd)
    unrestricted = calc.bond_orders(positions, chrg, **store)
    assert unrestricted.shape == (2, 2, 2)

    # closed shell: same total bond order and no magnetization
    assert pytest.approx(restricted.cpu()) == unrestricted[0].cpu()
    assert pytest.approx(0.0, abs=1e-10) == unrestricted[1].cpu()
