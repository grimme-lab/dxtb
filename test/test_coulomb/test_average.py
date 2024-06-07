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
Run tests for different averages in Coulomb kernel.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.components.interactions.coulomb import averaging_function
from dxtb._src.typing import DD
from ..conftest import DEVICE

a = torch.tensor([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_arithmetic(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    avg = averaging_function["arithmetic"](a.to(**dd))
    res = torch.tensor(
        [
            [1.000000, 1.500000, 2.000000, 2.500000],
            [1.500000, 2.000000, 2.500000, 3.000000],
            [2.000000, 2.500000, 3.000000, 3.500000],
            [2.500000, 3.000000, 3.500000, 4.000000],
        ],
        **dd,
    )

    assert pytest.approx(res.cpu()) == avg.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_geometric(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    avg = averaging_function["geometric"](a.to(**dd))
    res = torch.tensor(
        [
            [1.000000, 1.414214, 1.732051, 2.000000],
            [1.414214, 2.000000, 2.449490, 2.828427],
            [1.732051, 2.449490, 3.000000, 3.464102],
            [2.000000, 2.828427, 3.464102, 4.000000],
        ],
        **dd,
    )

    assert pytest.approx(res.cpu()) == avg.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_harmonic(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    avg = averaging_function["harmonic"](a.to(**dd))
    res = torch.tensor(
        [
            [1.000000, 1.333333, 1.500000, 1.600000],
            [1.333333, 2.000000, 2.400000, 2.666667],
            [1.500000, 2.400000, 3.000000, 3.428571],
            [1.600000, 2.666667, 3.428571, 4.000000],
        ],
        **dd,
    )

    assert pytest.approx(res.cpu()) == avg.cpu()
