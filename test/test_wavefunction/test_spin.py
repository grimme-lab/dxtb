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

from dxtb._src.typing import DD
from dxtb._src.wavefunction import spin

from ..conftest import DEVICE

sample_list = ["H2", "LiH", "SiH4"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
# @pytest.mark.parametrize("name", sample_list)
def test_updown_to_magnet(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    qsh_before = torch.tensor(
        [[-0.82824398, -0.33807717, -0.83367885], [0.0, 0.0, 0.0]],
        **dd,
    )

    qsh_after = spin.updown_to_magnet_2(qsh_before)

    ref_qsh_after = torch.tensor(
        [
            [-0.82824398, -0.33807717, -0.83367885],
            [-0.82824398, -0.33807717, -0.83367885],
        ],
        **dd,
    )
    assert (
        pytest.approx(ref_qsh_after.cpu(), rel=1e-7, abs=tol) == qsh_after.cpu()
    )


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_updown_to_magnet_scf_charge_layout(dtype: torch.dtype) -> None:
    """Validate alpha/beta -> charge/magnetization conversion on (..., 2, nao)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # Keep multiple axes with length 2 to ensure we convert exactly along -2.
    q_ab = torch.randn((2, 2, 2), **dd)
    q_ref = q_ab.clone()

    q_cm = spin.updown_to_magnet_2(q_ab)

    q_total_ref = q_ref[..., 0, :] + q_ref[..., 1, :]
    q_mag_ref = q_ref[..., 0, :] - q_ref[..., 1, :]

    assert torch.allclose(q_cm[..., 0, :], q_total_ref)
    assert torch.allclose(q_cm[..., 1, :], q_mag_ref)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_magnet_to_updown_scf_hamiltonian_layout(dtype: torch.dtype) -> None:
    """Validate charge/magnetization -> alpha/beta on (..., 2, nao, nao)."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # All axes are length 2 to stress dimension selection robustness.
    h_cm = torch.randn((2, 2, 2, 2), **dd)
    h_cm_ref = h_cm.clone()

    h_ab = h_cm.movedim(-3, -1).clone()
    h_ab = spin.magnet_to_updown(h_ab)
    h_ab = h_ab.movedim(-1, -3)

    h_alpha_ref = 0.5 * (h_cm_ref[..., 0, :, :] + h_cm_ref[..., 1, :, :])
    h_beta_ref = 0.5 * (h_cm_ref[..., 0, :, :] - h_cm_ref[..., 1, :, :])

    assert torch.allclose(h_ab[..., 0, :, :], h_alpha_ref)
    assert torch.allclose(h_ab[..., 1, :, :], h_beta_ref)

    # Roundtrip should recover the original charge/magnetization channels.
    h_cm_rt = h_ab.movedim(-3, -1).clone()
    h_cm_rt = spin.updown_to_magnet(h_cm_rt)
    h_cm_rt = h_cm_rt.movedim(-1, -3)

    assert torch.allclose(h_cm_rt, h_cm_ref)
