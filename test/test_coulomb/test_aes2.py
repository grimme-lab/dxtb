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
Test all multipole energy and potential contributions.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb import GFN2_XTB, IndexHelper
from dxtb._src.components.interactions.coulomb import new_aes2
from dxtb._src.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_energy(dtype: torch.dtype) -> None:
    """Test ES2 for some samples from MB16_43."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -1.50796743897235],
            [+0.00000000000000, +0.00000000000000, +1.50796743897235],
        ],
        **dd,
    )

    aes = new_aes2(torch.unique(numbers), GFN2_XTB, **dd)
    assert aes is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)
    cache = aes.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    # charges
    qat = torch.tensor([0.54699448343345114, -0.54699448343345114], **dd)
    qdp = torch.tensor(
        [
            [0.0000000000000000, 0.0000000000000000, -1.1260506806881299],
            [0.0000000000000000, 0.0000000000000000, 7.9884324667409912e-002],
        ],
        **dd,
    )
    qqp = torch.tensor(
        [
            [
                1.4096150819303312,
                0.0000000000000000,
                1.4096150819303312,
                0.0000000000000000,
                0.0000000000000000,
                -2.8192301638606625,
            ],
            [
                -2.3636549459148497e-003,
                0.0000000000000000,
                -2.3636549459148497e-003,
                0.0000000000000000,
                0.0000000000000000,
                4.7273098918297410e-003,
            ],
        ],
        **dd,
    )

    # Dipole energy

    ref_eat_dp = torch.tensor(
        [2.2503513865961954e-003, 1.4865828898185342e-004], **dd
    )
    eat_dp = aes.get_dipole_atom_energy(cache, qat=qat, qdp=qdp, qqp=qqp)
    assert pytest.approx(ref_eat_dp.cpu(), abs=tol) == eat_dp.cpu()

    # Quadrupole energy

    ref_eat_qp = torch.tensor(
        [8.8356975916636538e-003, 1.0826758104690651e-005], **dd
    )
    eat_qp = aes.get_quadrupole_atom_energy(cache, qat=qat, qdp=qdp, qqp=qqp)
    assert pytest.approx(ref_eat_qp.cpu(), abs=tol) == eat_qp.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_potential(dtype: torch.dtype) -> None:
    """Test ES2 for some samples from MB16_43."""
    tol = sqrt(torch.finfo(dtype).eps)
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -1.50796743897235],
            [+0.00000000000000, +0.00000000000000, +1.50796743897235],
        ],
        **dd,
    )

    aes = new_aes2(torch.unique(numbers), GFN2_XTB, **dd)
    assert aes is not None

    ihelp = IndexHelper.from_numbers(numbers, GFN2_XTB)
    cache = aes.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    # charges
    qat = torch.tensor([0.44132071106699799, -0.44132071106699783], **dd)
    qdp = torch.tensor(
        [
            [0.0000000000000000, 0.0000000000000000, -0.45042027227525194],
            [0.0000000000000000, 0.0000000000000000, 3.1953729866963966e-002],
        ],
        **dd,
    )
    qqp = torch.tensor(
        [
            [
                0.56384603277213252,
                0.0000000000000000,
                0.56384603277213252,
                0.0000000000000000,
                0.0000000000000000,
                -1.1276920655442650,
            ],
            [
                -9.4546197836593989e-004,
                0.0000000000000000,
                -9.4546197836593989e-004,
                0.0000000000000000,
                0.0000000000000000,
                1.8909239567318965e-003,
            ],
        ],
        **dd,
    )

    # Monopole potential

    ref_vat_mo = torch.tensor(
        [-4.1821215599096711e-004, -1.0724251999060196e-002], **dd
    )
    vat_mo = aes.get_monopole_atom_potential(cache, qat=qat, qdp=qdp, qqp=qqp)
    assert pytest.approx(ref_vat_mo.cpu(), abs=tol) == vat_mo.cpu()

    # Dipole potential

    ref_vat_dp = torch.tensor(
        [
            [0.0000000000000000, 0.0000000000000000, -1.6484335657207883e-003],
            [0.0000000000000000, 0.0000000000000000, +1.4390586834179975e-003],
        ],
        **dd,
    )
    vat_dp = aes.get_dipole_atom_potential(cache, qat=qat, qdp=qdp, qqp=qqp)
    assert pytest.approx(ref_vat_dp.cpu(), abs=tol) == vat_dp.cpu()

    # Quadrupole potential

    ref_vat_qp = torch.tensor(
        [
            [
                2.2553841310885301e-004,
                0.0000000000000000,
                2.2553841310885301e-004,
                0.0000000000000000,
                0.0000000000000000,
                -2.2973107195650411e-003,
            ],
            [
                -5.1869935057112189e-007,
                0.0000000000000000,
                -5.1869935057112189e-007,
                0.0000000000000000,
                0.0000000000000000,
                1.8472712920484780e-003,
            ],
        ],
        **dd,
    )
    vat_qp = aes.get_quadrupole_atom_potential(cache, qat=qat, qdp=qdp, qqp=qqp)
    assert pytest.approx(ref_vat_qp.cpu(), abs=tol) == vat_qp.cpu()
