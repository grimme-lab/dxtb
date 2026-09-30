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
Order and sign convention of the spherical components within a shell.

The pytorch integrals follow the CCA ordering of tblite (tblite/tblite#371):
ascending m for every shell, which is also the order of PySCF/libcint for
l >= 2. The p-orbitals are the exception: y, z, x (i.e., m = -1, 0, 1) here
and x, y, z in PySCF. Checked element-wise for all shell pairs up to f, which
energies and forces (invariant to the order within a shell) cannot detect.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.exlibs.available import has_pyscf
from dxtb._src.integral.driver.pytorch.impls.md import overlap_gto
from dxtb._src.typing import DD

# (number of primitives, principal quantum number, Slater exponent) per l
SHELLS = {0: (3, 1, 1.2), 1: (3, 2, 1.1), 2: (4, 3, 0.9), 3: (4, 4, 1.3)}

# PySCF order -> pytorch order within a shell
PERM_BY_L = {1: [1, 2, 0]}


@pytest.mark.skipif(has_pyscf is False, reason="PySCF not installed")
@pytest.mark.parametrize("la", [0, 1, 2, 3])
@pytest.mark.parametrize("lb", [0, 1, 2, 3])
def test_shell_pair_matches_pyscf(
    la: int, lb: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pylint: disable=import-outside-toplevel
    from pyscf import gto

    # dxtb's PySCF interface disables this globally on import
    monkeypatch.setattr(gto.mole, "NORMALIZE_GTO", True)

    dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}
    a = np.array([0.1, -0.2, 0.3])
    b = np.array([0.9, 0.4, -0.5])

    def cgto(l: int, norm: bool) -> tuple[torch.Tensor, torch.Tensor]:
        ng, n, zeta = SHELLS[l]
        return slater_to_gauss(ng, n, l, torch.tensor(zeta, **dd), norm=norm)

    (alpha_a, coeff_a), (alpha_b, coeff_b) = cgto(la, True), cgto(lb, True)
    s = overlap_gto(
        (torch.tensor(la), torch.tensor(lb)),
        (alpha_a, alpha_b),
        (coeff_a, coeff_b),
        torch.tensor(b - a, **dd),
    )

    # PySCF expects coefficients of normalized primitives (norm=False)
    basis = {
        sym: [[l] + [[float(x), float(c)] for x, c in zip(*cgto(l, False))]]
        for sym, l in (("Li", la), ("H", lb))
    }
    mol = gto.M(atom=[["Li", a], ["H", b]], basis=basis, unit="Bohr")
    ref = mol.intor("int1e_ovlp")[: 2 * la + 1, 2 * la + 1 :]
    ref = ref[PERM_BY_L.get(la, slice(None)), :]
    ref = ref[:, PERM_BY_L.get(lb, slice(None))]

    assert np.allclose(s.numpy(), ref, atol=1e-14, rtol=0.0)
