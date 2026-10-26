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
Test the spin constants of the spin-polarization interaction.

Reference values from tblite (``src/tblite/data/spin.f90``).
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.components.interactions.spin import (
    MAX_SPIN_ELEMENT,
    load_spin_constants,
)
from dxtb._src.typing import DD

from ..conftest import DEVICE

# columns: ss, sp, pp, sd, pd, dd
ref = {
    1: [-0.0716250, 0.0, 0.0, 0.0, 0.0, 0.0],
    6: [-0.0305000, -0.0250250, -0.0226750, 0.0, 0.0, 0.0],
    24: [
        -0.0144750,
        -0.0116120,
        -0.0160000,
        -0.0037250,
        -0.0014630,
        -0.0157750,
    ],
    71: [-0.1086250, -0.0079000, 0.0063250, -0.0047000, -0.0007120, -0.0269000],
    86: [
        -0.0139000,
        -0.0097380,
        -0.0106500,
        -0.0028750,
        -0.0078120,
        -0.0130000,
    ],
}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_table(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    table = load_spin_constants(**dd)

    assert table.shape == (MAX_SPIN_ELEMENT + 1, 6)
    assert table.dtype == dtype

    # padding
    assert (table[0] == 0.0).all()

    # all physical elements have (negative) ss constants
    assert (table[1:, 0] < 0.0).all()


@pytest.mark.parametrize("number", ref.keys())
def test_element(number: int) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    table = load_spin_constants(**dd)

    reference = torch.tensor(ref[number], **dd)
    assert pytest.approx(reference.cpu(), abs=1e-10) == table[number].cpu()
