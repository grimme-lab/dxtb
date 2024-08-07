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
Test orthogonality of GFN1-xTB's H1s and H2s orbitals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.basis.ortho import gaussian_integral, orthogonalize
from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.integral.driver.pytorch.impls.md import overlap_gto
from dxtb._src.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ortho_1s_2s(dtype: torch.dtype):
    """Test orthogonality of GFN1-xTB's H1s and H2s orbitals"""
    tols = {"abs": 1e-6, "rel": 1e-6, "nan_ok": False}
    dd: DD = {"dtype": dtype, "device": DEVICE}

    # azimuthal quantum number of s-orbital
    l = torch.tensor(0, device=DEVICE)

    # same site
    vec = torch.tensor([0.0, 0.0, 0.0], **dd)

    # create gaussians
    alphai, coeffi = slater_to_gauss(
        torch.tensor(5, device=DEVICE),
        torch.tensor(1, device=DEVICE),
        l,
        vec.new_tensor(1.2),
    )
    alphaj, coeffj = slater_to_gauss(
        torch.tensor(2, device=DEVICE),
        torch.tensor(2, device=DEVICE),
        l,
        vec.new_tensor(0.7),
    )

    alphaj_new, coeffj_new = orthogonalize((alphai, alphaj), (coeffi, coeffj))

    # normalised self-overlap
    ref = torch.tensor(1, **dd)
    s = overlap_gto((l, l), (alphaj, alphaj), (coeffj, coeffj), vec)
    assert pytest.approx(ref.cpu(), **tols) == s.sum().cpu()
    s2 = gaussian_integral(*(alphaj, alphaj), *(coeffj, coeffj))
    assert pytest.approx(ref.cpu(), **tols) == s2.cpu()

    # orthogonal overlap
    ref = torch.tensor(0, **dd)
    s = overlap_gto((l, l), (alphai, alphaj_new), (coeffi, coeffj_new), vec)
    s2 = gaussian_integral(*(alphai, alphaj_new), *(coeffi, coeffj_new))
    assert pytest.approx(ref.cpu(), **tols) == s.sum().cpu()
    assert pytest.approx(ref.cpu(), **tols) == s2.cpu()
