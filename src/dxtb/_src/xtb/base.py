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
xTB Hamiltonians: Base
======================

Base class for xTB Hamiltonians.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.param import Param
from dxtb._src.typing import Tensor, TensorLike

__all__ = ["HamiltonianABC", "BaseHamiltonian"]


class HamiltonianABC(ABC):
    """
    Abstract base class for Hamiltonians.
    """

    @abstractmethod
    def build(
        self, positions: Tensor, overlap: Tensor, cn: Tensor | None = None
    ) -> Tensor:
        """
        Build the xTB Hamiltonian.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of molecular structure.
        overlap : Tensor
            Overlap matrix.
        cn : Tensor | None, optional
            Coordination number. Defaults to ``None``.

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
            Atomic positions of molecular structure.
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


class BaseHamiltonian(HamiltonianABC, TensorLike):
    """
    Base class for GFN Hamiltonians.

    For the Hamiltonians, no integral driver is needed. Therefore, the
    signatures are different from the integrals over atomic orbitals. The most
    important difference is the `build` method, which does not require the
    driver anymore and only takes the positions (and the overlap integral).
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""
    unique: Tensor
    """Unique species of the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    hscale: Tensor
    """Off-site scaling factor for the Hamiltonian."""
    kcn: Tensor
    """Coordination number dependent shift of the self energy."""
    kpair: Tensor
    """Element-pair-specific parameters for scaling the Hamiltonian."""
    refocc: Tensor
    """Reference occupation numbers."""
    selfenergy: Tensor
    """Self-energy of each species."""
    shpoly: Tensor
    """Polynomial parameters for the distant dependent scaling."""
    valence: Tensor
    """Whether the shell belongs to the valence shell."""

    en: Tensor
    """Pauling electronegativity of each species."""
    rad: Tensor
    """Van-der-Waals radius of each species."""

    __slots__ = [
        "numbers",
        "unique",
        "par",
        "ihelp",
        "hscale",
        "kcn",
        "kpair",
        "refocc",
        "selfenergy",
        "shpoly",
        "valence",
        "en",
        "rad",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_,
    ) -> None:
        super().__init__(device, dtype)

        # check device of input tensors
        if any(tensor.device != self.device for tensor in (numbers, ihelp)):
            raise ValueError("All input tensors must be on the same device")

        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp

        self.label = self.__class__.__name__
        self._matrix = None

    @property
    def matrix(self) -> Tensor | None:
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    def get_occupation(self) -> Tensor:
        """
        Obtain the reference occupation numbers for each orbital.
        """

        refocc = self.ihelp.spread_ushell_to_orbital(self.refocc)
        orb_per_shell = self.ihelp.spread_shell_to_orbital(
            self.ihelp.orbitals_per_shell
        )

        return torch.where(
            orb_per_shell != 0,
            refocc / orb_per_shell,
            torch.tensor(0, **self.dd),
        )
