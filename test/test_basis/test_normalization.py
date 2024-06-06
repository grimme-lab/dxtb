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
Test CGTO normalization.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.integral.driver.pytorch.impls.md import overlap_gto
from ..conftest import DEVICE
from dxtb._src.typing import DD


@pytest.mark.parametrize("ng", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "n, l",
    [
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (3, 2),
        (4, 2),
        (5, 2),
        (4, 3),
        (5, 3),
        # (5, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sto_ng_single(ng, n, l, dtype):
    """
    Test normalization of all STO-NG basis functions
    """
    dd: DD = {"dtype": dtype, "device": DEVICE}
    atol = 1.0e-6 if dtype == torch.float else 2.0e-7

    alpha, coeff = slater_to_gauss(ng, n, l, torch.tensor(1.0, **dd))
    angular = torch.tensor(l, device=DEVICE)
    vec = torch.zeros((3,), **dd)

    s = overlap_gto((angular, angular), (alpha, alpha), (coeff, coeff), vec)
    ref = torch.diag(torch.ones((2 * l + 1,), **dd))

    assert pytest.approx(ref.cpu(), abs=atol) == s.cpu()


@pytest.mark.parametrize("ng", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sto_ng_batch(ng: int, dtype: torch.dtype):
    """
    Test symmetry of s integrals
    """
    dd: DD = {"dtype": dtype, "device": DEVICE}

    n, l = torch.tensor(1, device=DEVICE), torch.tensor(0, device=DEVICE)
    ng_ = torch.tensor(ng, device=DEVICE)

    coeff, alpha = slater_to_gauss(ng_, n, l, torch.tensor(1.0, **dd))
    coeff, alpha = coeff.type(dtype)[:ng_], alpha.type(dtype)[:ng_]
    vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        **dd,
    )

    s = overlap_gto((l, l), (alpha, alpha), (coeff, coeff), vec)

    assert pytest.approx(s[0, :].cpu()) == s[1, :].cpu()
    assert pytest.approx(s[0, :].cpu()) == s[2, :].cpu()


@pytest.mark.parametrize("ng", [6])
@pytest.mark.parametrize("n, l", [(1, 0)])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_no_norm(ng, n, l, dtype):
    """
    Test normalization of all STO-NG basis functions
    """
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = 1.0e-7

    ref_alpha = torch.tensor(
        [
            23.1030311584,
            4.2359156609,
            1.1850565672,
            0.4070988894,
            0.1580884159,
            0.0651095361,
        ],
        **dd,
    )

    ref_coeff = torch.tensor(
        [
            0.0091635967,
            0.0493614934,
            0.1685383022,
            0.3705627918,
            0.4164915383,
            0.1303340793,
        ],
        **dd,
    )

    zeta = torch.tensor(1.0, **dd)
    alpha, coeff = slater_to_gauss(ng, n, l, zeta, norm=False)

    assert pytest.approx(ref_alpha.cpu(), abs=tol, rel=tol) == alpha.cpu()
    assert pytest.approx(ref_coeff.cpu(), abs=tol, rel=tol) == coeff.cpu()
