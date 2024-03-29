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
PyTorch-based overlap implementations.
"""

from __future__ import annotations

import torch

from ....._types import Literal, Tensor
from .....basis import Basis, IndexHelper
from .....constants import defaults
from .....utils import batch
from .md import overlap_gto

__all__ = ["overlap_legacy", "overlap_gradient_legacy"]


def overlap_legacy(
    positions: Tensor,
    bas: Basis,
    ihelp: IndexHelper,
    uplo: Literal["n", "u", "l"] = "l",
    cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
) -> Tensor:
    """
    Calculate the full overlap matrix.

    The argument `uplo` has no effect. The legacy code always computes the
    upper triangular matrix and mirrors it to the lower part.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    bas : Basis
        Basis set information.
    ihelp : IndexHelper
        Helper class for indexing.
    uplo : Literal['n';, 'u', 'l'], optional
        Whether the matrix of unique shell pairs should be create as a
        triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
        Defaults to `l` (lower triangular matrix).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff for integral calculation in Bohr. Defaults to
        `constants.defaults.INTCUTOFF`.

    Returns
    -------
    Tensor
        Orbital-resolved overlap matrix of shape `(nb, norb, norb)`.
    """
    # create empty overlap matrix
    overlap = torch.zeros(ihelp.nao, ihelp.nao)

    # Create alphas and sort for indexing
    alphas, coeffs = bas.create_cgtos()
    alpha = ihelp.spread_ushell_to_shell(batch.pack(alphas), dim=-2, extra=True)
    coeff = ihelp.spread_ushell_to_shell(batch.pack(coeffs), dim=-2, extra=True)

    for iat in range(positions.shape[-2]):
        for jat in range(iat):
            vec = positions[iat, :] - positions[jat, :]

            if torch.norm(vec) > cutoff:
                continue

            for ish in range(ihelp.shells_per_atom[iat]):
                sii = ihelp.shell_index[iat] + ish
                angi = ihelp.angular[sii]
                ii = ihelp.orbital_index[sii]
                i_nao = 2 * angi + 1

                for jsh in range(ihelp.shells_per_atom[jat]):
                    sij = ihelp.shell_index[jat] + jsh
                    angj = ihelp.angular[sij]
                    jj = ihelp.orbital_index[sij]
                    j_nao = 2 * angj + 1

                    a = (batch.deflate(alpha[sii]), batch.deflate(alpha[sij]))
                    c = (batch.deflate(coeff[sii]), batch.deflate(coeff[sij]))
                    l = (angi, angj)

                    stmp = overlap_gto(l, a, c, -vec)

                    for iao in range(i_nao):
                        for jao in range(j_nao):
                            overlap[jj + jao, ii + iao] = stmp[iao, jao]

    # fill empty triangular matrix
    overlap = torch.triu(overlap, diagonal=1) + torch.tril(overlap.mT)

    # fix diagonal as "self-overlap" was removed via loops
    overlap.fill_diagonal_(1.0)

    return overlap


def overlap_gradient_legacy(
    positions: Tensor,
    bas: Basis,
    ihelp: IndexHelper,
    uplo: Literal["n", "u", "l"] = "l",
    cutoff: Tensor | float | int | None = None,
) -> Tensor:
    """
    Calculate the gradient of the overlap.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    bas : Basis
        Basis set information.
    ihelp : IndexHelper
        Helper class for indexing.
    uplo : Literal['n';, 'u', 'l'], optional
        Whether the matrix of unique shell pairs should be create as a
        triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
        Defaults to `l` (lower triangular matrix).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff for integral calculation in Angstrom. Defaults to
        `constants.defaults.INTCUTOFF` (50.0).

    Returns
    -------
    Tensor
        Orbital-resolved overlap gradient of shape `(nb, norb, norb, 3)`.
    """
    raise NotImplementedError(
        "The legacy overlap code has no explicit analytical gradient routine."
    )
