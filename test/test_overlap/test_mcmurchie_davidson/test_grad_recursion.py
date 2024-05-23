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
Run tests for overlap gradient.
"""

from __future__ import annotations

import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck

from dxtb._src.integral.driver.pytorch.impls.md import recursion


def test_gradcheck_efunction(dtype: torch.dtype = torch.double) -> None:
    xij = torch.tensor(
        [
            [0.0328428932, 0.0555253364, 0.0625081286, 0.0645959303],
            [0.0555253364, 0.1794814467, 0.2809201777, 0.3286595047],
            [0.0625081286, 0.2809201777, 0.6460560560, 0.9701334238],
            [0.0645959303, 0.3286595047, 0.9701334238, 1.9465910196],
        ],
        dtype=dtype,
    )
    rpi = torch.tensor(
        [
            [
                [
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                ],
                [
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                ],
                [
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                ],
            ]
        ],
        dtype=dtype,
    )
    rpj = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        dtype=dtype,
    )
    shape = (1, 3, 4, 4, torch.tensor(1), torch.tensor(1))

    assert dgradcheck(
        recursion.EFunction.apply,  # type: ignore
        (xij.clone().requires_grad_(True), rpi, rpj, shape),  # type: ignore
    )
    assert dgradgradcheck(
        recursion.EFunction.apply,  # type: ignore
        (xij.clone().requires_grad_(True), rpi, rpj, shape),  # type: ignore
    )

    assert dgradcheck(
        recursion.EFunction.apply,  # type: ignore
        (
            xij,
            rpi.clone().requires_grad_(True),
            rpj.clone().requires_grad_(True),
            shape,
        ),  # type: ignore
    )
    assert dgradgradcheck(
        recursion.EFunction.apply,  # type: ignore
        (
            xij,
            rpi.clone().requires_grad_(True),
            rpj.clone().requires_grad_(True),
            shape,
        ),  # type: ignore
    )
