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
Element-wise agreement of the overlap of all pytorch drivers with libcint.

The pytorch drivers follow the CCA ordering of tblite (spherical components
in ascending m), which libcint shares for l >= 2. Only the p-orbitals differ:
y, z, x in the pytorch drivers and x, y, z in libcint. Energies and forces are
invariant to this fixed permutation, so the comparison is done after
reordering.

Guards the double precision of the cartesian-to-spherical transformation
matrices and the exact normalization of the contracted shells; either would
show up as deviations of up to ~5e-9 for d shells (SiH4).
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.constants.labels import (
    INTDRIVER_LIBCINT,
    INTDRIVER_PYTORCH,
)
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.pytorch.impls.trafo import TRAFO
from dxtb._src.typing import DD, Tensor
from dxtb.integrals.wrappers import overlap

from ..conftest import DEVICE
from .samples import samples

# permutation of within-shell components (libcint -> pytorch), indexed by l
PERM_BY_L = {
    0: [0],
    1: [1, 2, 0],
    2: [0, 1, 2, 3, 4],
}


def permutation_matrix(ihelp: IndexHelper, dd: DD) -> Tensor:
    """AO permutation matrix mapping the libcint to the pytorch order."""
    angular = ihelp.angular.tolist()
    orbs_per_shell = ihelp.orbitals_per_shell.tolist()

    n = sum(orbs_per_shell)
    p = torch.zeros((n, n), **dd)

    off = 0
    for l, nao in zip(angular, orbs_per_shell):
        for k, pk in enumerate(PERM_BY_L[l]):
            p[off + k, off + pk] = 1.0
        off += nao

    return p


def test_trafo_is_double() -> None:
    """Irrational coefficients (sqrt(3), ...) must not be truncated to the
    default dtype at import."""
    for trafo in TRAFO:
        assert trafo.dtype == torch.double

    assert TRAFO[2][1, 4].item() == 3.0**0.5


@pytest.mark.skipif(
    has_libcint is False, reason="libcint interface not installed"
)
@pytest.mark.parametrize(
    "name", ["H2", "LiH", "CH4", "NH3", "SiH4", "LYS_xao_dist"]
)
def test_overlap_matches_libcint(name: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    s_lib = overlap(numbers, positions, par, driver=INTDRIVER_LIBCINT)
    s_pt = overlap(numbers, positions, par, driver=INTDRIVER_PYTORCH)

    p = permutation_matrix(ihelp, dd)
    s_lib = p @ s_lib.to(DEVICE) @ p.mT

    assert torch.allclose(s_pt, s_lib, atol=1e-12, rtol=0.0)
