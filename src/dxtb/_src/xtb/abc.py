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
xTB Hamiltonians: ABC
=====================

Abstract case class for xTB Hamiltonians.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dxtb._src.components.interactions import Potential
    from dxtb._src.param.module import ParamModule
    from dxtb._src.typing import Tensor

__all__ = ["HamiltonianABC"]


class HamiltonianABC(ABC):
    """
    Abstract base class for Hamiltonians.
    """

    @abstractmethod
    def _get_hscale(self, par: ParamModule) -> Tensor:
        """
        Obtain the off-site scaling factor for the Hamiltonian.

        Parameters
        ----------
        par : ParamModule
            Representation of an extended tight-binding model.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """

    @abstractmethod
    def _get_elem_valence(self, par: ParamModule) -> Tensor:
        """
        Obtain a mask for valence and non-valence shells. This is only required
        for GFN1-xTB's second hydrogen s-function.

        Parameters
        ----------
        par : ParamModule
            Representation of an extended tight-binding model.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each unique species.
        """

    @abstractmethod
    def build(self, positions: Tensor, overlap: Tensor | None = None) -> Tensor:
        """
        Build the xTB Hamiltonian.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        overlap : Tensor | None, optional
            Overlap matrix. If ``None``, the true xTB Hamiltonian is *not*
            built. Defaults to ``None``.

        Returns
        -------
        Tensor
            Hamiltonian (always symmetric).
        """

    @abstractmethod
    def get_gradient(
        self,
        positions: Tensor,
        overlap: Tensor,
        doverlap: Tensor,
        pmat: Tensor,
        wmat: Tensor,
        pot: Potential,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate gradient of the full Hamiltonian with respect ot atomic positions.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``)..
        overlap : Tensor
            Overlap matrix.
        doverlap : Tensor
            Derivative of the overlap matrix.
        pmat : Tensor
            Density matrix.
        wmat : Tensor
            Energy-weighted density.
        pot : Tensor
            Self-consistent electrostatic potential.
        cn : Tensor
            Coordination number.

        Returns
        -------
        tuple[Tensor, Tensor]
            Derivative of energy with respect to coordination number (first
            tensor) and atomic positions (second tensor).
        """
