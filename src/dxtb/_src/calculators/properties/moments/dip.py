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
Properties: Dipole Moment
=========================

Analytical calculation of dipole moment.

This module serves more as a short-cut for the calculation in
:class:`~dxtb.Calculator`, hiding some implementation details.
"""

from __future__ import annotations

from tad_mctc.data.getters import get_zvalence
from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

__all__ = ["dipole", "dipole_xtb"]


def dipole(
    charge: Tensor, positions: Tensor, density: Tensor, integral: Tensor
) -> Tensor:
    r"""
    Analytical calculation of electric dipole moment with electric dipole
    contribution from nuclei (:math:`\sum_i(r_{ik}  q_i)`) and electrons.

    Parameters
    ----------
    charge : Tensor
        Atom-resolved charges.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    density : Tensor
        Density matrix.
    integral : Tensor
        Dipole integral.

    Returns
    -------
    Tensor
        Electric dipole moment.

    Note
    ----
    This version follows the `tblite` implementation, which employs ``r-rj`` as
    moment operator and requires the SCC charges for the nuclear dipole
    contribution.
    """
    # TODO: Shape checks

    e_dipole = -einsum("...xij,...ij->...x", integral, density)
    n_dipole = einsum("...ix,...i->...x", positions, charge)
    return n_dipole + e_dipole


def dipole_xtb(
    numbers: Tensor, positions: Tensor, density: Tensor, integral: Tensor
) -> Tensor:
    r"""
    Analytical calculation of electric dipole moment with electric dipole
    contribution from nuclei (:math:`\sum_i(r_{ik}  q_i)`) and electrons.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    charge : Tensor
        Atom-resolved charges.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    density : Tensor
        Density matrix.
    integral : Tensor
        Dipole integral.

    Returns
    -------
    Tensor
        Electric dipole moment.

    Note
    ----
    This version follows the `xtb` implementation, where the ``r0`` moment
    operator is used and the nuclear contribution uses the valence charges.
    """
    # TODO: Shape checks

    # electric component from dipole integral and density matrix
    e_dipole = -einsum("...xij,...ij->...x", integral, density)

    # moment operator "r0" combines with valence charges (xtb implementation)
    n_dipole = einsum(
        "...ix,...i->...x",
        positions,
        get_zvalence(numbers, device=positions.device, dtype=positions.dtype),
    )

    dip = n_dipole + e_dipole
    return dip
