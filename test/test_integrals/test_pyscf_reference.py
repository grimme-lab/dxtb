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
Overlap against PySCF (independent of dxtb's own libcint wrapper).

Normalization: ``Basis.create_cgtos()`` already includes the standard
per-primitive Cartesian-Gaussian normalization in the contraction
coefficients, but PySCF's basis loader applies it again. The coefficients are
therefore divided by ``N(l, a) = (2a/pi)^0.75 * (4a)^(l/2) / sqrt((2l-1)!!)``
before the PySCF basis is built; otherwise every primitive is double
normalized, which breaks the Gram-Schmidt-orthogonalized double-zeta shells
of H.

``dxtb._src.exlibs.pyscf.mol.base`` sets ``pyscf.gto.mole.NORMALIZE_GTO =
False`` as an import-time side effect. This file relies on PySCF's default
(``True``), so it saves and restores the flag around every ``gto.M`` call;
without that the result depends on the test order within a worker.

The pytorch drivers and PySCF agree to ``1e-12`` and better, also for the
d shells of SiH4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import factorial2

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.pytorch.impls.kernels import (
    ALGORITHMS,
    get_kernel,
)
from dxtb._src.typing import DD
from dxtb.integrals.wrappers import overlap
from dxtb.labels import INTDRIVER_LIBCINT, INTDRIVER_PYTORCH

from ..conftest import DEVICE
from .samples import samples

try:
    from pyscf import gto
    from pyscf.data import elements

    has_pyscf = True
except ImportError:
    has_pyscf = False

pytestmark = pytest.mark.skipif(
    not (has_pyscf and has_libcint), reason="pyscf or libcint not available"
)

PERM_BY_L = {0: [0], 1: [1, 2, 0], 2: [0, 1, 2, 3, 4]}


def _prim_norm(l: int, a: float) -> float:
    """Standard primitive Cartesian-Gaussian normalization (stretched, x^l)."""
    denom = math.sqrt(factorial2(2 * l - 1)) if l > 0 else 1.0
    return (2 * a / math.pi) ** 0.75 * (4 * a) ** (l / 2) / denom


def _build_pyscf_mol(numbers: torch.Tensor, positions: torch.Tensor):
    ihelp = IndexHelper.from_numbers(numbers.cpu(), par)
    dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}
    bas = Basis(numbers.cpu(), par, ihelp, **dd)
    alphas, coeffs = bas.create_cgtos()

    shells_to_ushell = ihelp.shells_to_ushell.tolist()
    shells_to_atom = ihelp.shells_to_atom.tolist()
    angular = ihelp.angular.tolist()
    pos = positions.cpu().tolist()

    atom_lines = []
    basis_dict: dict[str, list] = {}
    for i, z in enumerate(numbers.cpu().tolist()):
        sym = f"{elements.ELEMENTS[z]}{i}"
        atom_lines.append((sym, tuple(pos[i])))

        shell_idxs = [s for s, at in enumerate(shells_to_atom) if at == i]
        entries = []
        for s in shell_idxs:
            ush = shells_to_ushell[s]
            l = angular[s]
            a, c = alphas[ush], coeffs[ush]
            unnorm_c = [
                ci / _prim_norm(l, ai) for ai, ci in zip(a.tolist(), c.tolist())
            ]
            entries.append(
                [l]
                + [
                    [float(ai), float(ci)]
                    for ai, ci in zip(a.tolist(), unnorm_c)
                ]
            )
        basis_dict[sym] = entries

    spin = int(numbers.sum().item()) % 2

    # `dxtb._src.exlibs.pyscf` (imported elsewhere in the suite, e.g. by
    # test/test_mol/test_external.py) flips this process-global PySCF flag
    # to False as an import-time side effect. This harness relies on
    # PySCF's default (True) normalization, so save and restore it
    # explicitly rather than assume the library default holds.
    prev_normalize = gto.mole.NORMALIZE_GTO
    gto.mole.NORMALIZE_GTO = True
    try:
        return gto.M(atom=atom_lines, basis=basis_dict, unit="Bohr", spin=spin)
    finally:
        gto.mole.NORMALIZE_GTO = prev_normalize


@pytest.mark.parametrize("name", ["LiH", "SiH4"])
def test_overlap_matches_pyscf_via_libcint_order(name: str) -> None:
    """PySCF's own AO order matches dxtb's libcint driver directly."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    mol = _build_pyscf_mol(numbers, positions)
    s_pyscf = mol.intor("int1e_ovlp")

    s_lib = overlap(numbers, positions, par, driver=INTDRIVER_LIBCINT).to(
        DEVICE
    )
    s_lib = s_lib.detach().cpu().numpy()

    assert np.abs(s_pyscf - s_lib).max() < 1e-13


def test_pytorch_matches_libcint_and_pyscf() -> None:
    """
    The pytorch driver matches both libcint and an independent PySCF
    reference to machine precision on SiH4 (d shells), in the same AO order.
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    angular = ihelp.angular.tolist()
    orbs = ihelp.orbitals_per_shell.tolist()

    n = sum(orbs)
    p = torch.zeros((n, n), **dd)
    off = 0
    for l, nao in zip(angular, orbs):
        perm = PERM_BY_L[l]
        for k, pk in enumerate(perm):
            p[off + k, off + pk] = 1.0
        off += nao

    mol = _build_pyscf_mol(numbers, positions)
    s_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).to(**dd)
    s_pyscf_reordered = p @ s_pyscf @ p.mT

    s_pt = overlap(numbers, positions, par, driver=INTDRIVER_PYTORCH)
    s_lib = overlap(numbers, positions, par, driver=INTDRIVER_LIBCINT).to(
        DEVICE
    )
    s_lib_reordered = p @ s_lib @ p.mT

    diff_vs_pyscf = (s_pt - s_pyscf_reordered).abs()
    diff_vs_libcint = (s_pt - s_lib_reordered).abs()

    # both independent references disagree with pytorch by essentially the
    # same amount -- confirms the residual lives in the pytorch driver.
    assert torch.allclose(diff_vs_pyscf, diff_vs_libcint, atol=1e-12, rtol=0.0)

    mask = s_pt.abs() > 1e-6
    assert diff_vs_pyscf[mask].max() < 1e-12


@pytest.mark.parametrize("name", ["LiH", "SiH4"])
@pytest.mark.parametrize("kind", ["dipole", "quadrupole"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_os_multipoles_match_pyscf(
    name: str, kind: str, algorithm: str
) -> None:
    """
    Dipole / quadrupole of every kernel against PySCF's ``int1e_r`` /
    ``int1e_rr`` with the common origin at (0, 0, 0): independent of dxtb's
    libcint wrapper.
    """
    # pylint: disable=import-outside-toplevel
    from dxtb._src.integral.driver.pytorch.impls.pairs import assemble_matrix
    from dxtb._src.integral.driver.pytorch.impls.pipeline import (
        DIPOLE_COMPONENTS,
        QUADRUPOLE_COMPONENTS,
    )

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    alphas, coeffs = Basis(numbers, par, ihelp, **dd).create_cgtos()
    comps = DIPOLE_COMPONENTS if kind == "dipole" else QUADRUPOLE_COMPONENTS
    mine = assemble_matrix(
        get_kernel(algorithm), ihelp, alphas, coeffs, positions, comps
    )

    mol = _build_pyscf_mol(numbers, positions)
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ref = mol.intor(
            "int1e_r" if kind == "dipole" else "int1e_rr", comp=len(comps)
        )
    ref = torch.from_numpy(ref).to(**dd)

    angular = ihelp.angular.tolist()
    orbs = ihelp.orbitals_per_shell.tolist()
    n = sum(orbs)
    p = torch.zeros((n, n), **dd)
    off = 0
    for l, nao in zip(angular, orbs):
        for k, pk in enumerate(PERM_BY_L[l]):
            p[off + k, off + pk] = 1.0
        off += nao

    assert torch.allclose(mine, p @ ref @ p.mT, atol=1e-12, rtol=0.0)
