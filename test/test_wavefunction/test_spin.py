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
Test for changing the representation of spin-polarized densities.
Reference values are calculated using tblite version 0.5.0
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.typing import DD
from dxtb._src.wavefunction import spin
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian

from ..conftest import DEVICE
from .samples import samples

sample_list = ["H2", "LiH", "SiH4"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
# @pytest.mark.parametrize("name", sample_list)
def test_updown_to_magnet(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    qsh_before = torch.tensor(
        [[-0.82824398, -0.33807717, -0.83367885], [0.0, 0.0, 0.0]]
    )

    qsh_after = spin.updown_to_magnet_2(qsh_before)

    ref_qsh_after = torch.tensor(
        [
            [-0.82824398, -0.33807717, -0.83367885],
            [-0.82824398, -0.33807717, -0.83367885],
        ]
    )
    assert (
        pytest.approx(ref_qsh_after.cpu(), rel=1e-7, abs=tol) == qsh_after.cpu()
    )
