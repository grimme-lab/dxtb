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
SCF: Result
===========

Result type for SCF.
"""
from __future__ import annotations

from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.typing import Tensor, TypedDict

__all__ = ["SCFResult"]


class SCFResult(TypedDict):
    """Collection of SCF result variables."""

    charges: Charges
    """Self-consistent orbital-resolved Mulliken partial charges."""

    coefficients: Tensor
    """LCAO-MO coefficients (eigenvectors of Fockian)."""

    density: Tensor
    """Density matrix."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    energy: Tensor
    """Energies of the self-consistent contributions (interactions)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    occupation: Tensor
    """Orbital occupation."""

    potential: Potential
    """Self-consistent orbital-resolved potential."""

    iterations: int
    """Number of SCF iterations."""
