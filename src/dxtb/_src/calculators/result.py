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
Calculators: Result
===================

Result container for singlepoint calculation.
"""

from __future__ import annotations

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Any, Tensor, TensorLike

__all__ = ["Result"]


class Result(TensorLike):
    """
    Result container for singlepoint calculation.
    """

    charges: Charges
    """Self-consistent orbital-resolved Mulliken partial charges."""

    coefficients: Tensor
    """LCAO-MO coefficients (eigenvectors of Fockian)."""

    density: Tensor
    """Density matrix."""

    cenergies: dict[str, Tensor]
    """Energies of classical contributions."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    integrals: IntegralMatrices
    """Collection of integrals including overlap and core Hamiltonian (H0)."""

    occupation: Tensor
    """Orbital occupation."""

    potential: Potential
    """Self-consistent potentials."""

    scf: Tensor
    """Atom-resolved energy from the self-consistent field (SCF) calculation."""

    total: Tensor
    """Total energy."""

    __slots__ = [
        "charges",
        "coefficients",
        "cenergies",
        "density",
        "emo",
        "fenergy",
        "hamiltonian",
        "integrals",
        "iter",
        "occupation",
        "potential",
        "scf",
        "total",
    ]

    def __init__(
        self,
        positions: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        shape = positions.shape[:-1]

        self.scf = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.fenergy = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.total = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.cenergies = {}
        self.iter = 0

    def __str__(self) -> str:
        """Custom print representation showing all available slots."""
        return f"{self.__class__.__name__}({self.__slots__})"

    def __repr__(self) -> str:
        """Custom print representation showing all available slots."""
        return str(self)

    def get_energies(self) -> dict[str, dict[str, Any]]:
        """
        Get energies in a dictionary.

        Returns
        -------
        dict[str, dict[str, float]]
            Energies in a dictionary.
        """
        KEY = "value"

        c = {k: {KEY: v.sum().item()} for k, v in self.cenergies.items()}
        ctotal = sum(d[KEY] for d in c.values())

        e = {
            "SCF": {KEY: self.scf.sum().item()},
            "Free Energy (Fermi)": {KEY: self.fenergy.sum().item()},
        }
        etotal = sum(d[KEY] for d in e.values())

        return {
            "total": {KEY: self.total.sum().item()},
            "Classical": {KEY: ctotal, "sub": c},
            "Electronic": {KEY: etotal, "sub": e},
        }

    def print_energies(
        self, v: int = 4, precision: int = 14
    ) -> None:  # pragma: no cover
        """Print energies in a table."""

        OutputHandler.write_table(
            self.get_energies(),
            title="Energies",
            columns=["Contribution", "Energy (Eh)"],
            v=v,
            precision=precision,
        )
