# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Non-self-consistent SCF solver
================================
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.components.interactions import (
    Charges,
    InteractionList,
    InteractionListCache,
    Potential,
)
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Any, Tensor
from dxtb.config import ConfigSCF

from .pure.conversions import density_to_charges, hamiltonian_to_density
from .pure.data import _Data
from .pure.energies import get_electronic_free_energy, get_energy
from .result import SCFResult

__all__ = ["solve_nonselfconsistent"]


def _zero_potential(charges: Charges, batch_mode: int) -> Potential:
    """Construct a zero potential with shapes matching all available charges."""
    return Potential(
        mono=torch.zeros_like(charges.mono),
        dipole=(
            torch.zeros_like(charges.dipole)
            if charges.dipole is not None
            else None
        ),
        quad=(
            torch.zeros_like(charges.quad) if charges.quad is not None else None
        ),
        batch_mode=batch_mode,
    )


def solve_nonselfconsistent(
    numbers: Tensor,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    n0: Tensor,
    occupation: Tensor,
    **kwargs: Any,
) -> SCFResult:
    """
    Solve a non-self-consistent electronic problem with exactly one eigensolve.

    The initial GFN0 implementation is deliberately energy-only with respect
    to density-dependent interactions. Applying one arbitrary interaction pass
    would not be a well-defined physical model, so such components are rejected.
    """
    if len(interactions.components) > 0:
        interaction_labels = ", ".join(
            interaction.label for interaction in interactions.components
        )
        raise ValueError(
            "Non-self-consistent GFN0 does not support charge-dependent "
            f"Interaction components (received: {interaction_labels})."
        )

    data = _Data(
        occupation=occupation,
        n0=n0,
        numbers=numbers,
        ihelp=ihelp,
        cache=cache,
        integrals=integrals,
    )

    config.eigen_options = {
        "method": "exacteig",
        **kwargs.pop("eigen_options", {}),
    }

    # H0 is the full Hamiltonian for the non-self-consistent method.
    data.hamiltonian = integrals.hcore

    # This helper performs the sole generalized H0/S diagonalization, optional
    # Fermi filling, and density construction.
    data.density = hamiltonian_to_density(
        data.hamiltonian,
        data,
        config,
    )

    # Mulliken analysis also constructs the orbital band-energy partition.
    charges = density_to_charges(data.density, data, config)
    charges.nullify_padding()

    energy = get_energy(charges, data, interactions)
    fenergy = get_electronic_free_energy(data, config)

    return {
        "charges": charges,
        "coefficients": data.evecs,
        "density": data.density,
        "emo": data.evals,
        "energy": energy,
        "fenergy": fenergy,
        "hamiltonian": data.hamiltonian,
        "occupation": data.occupation,
        "potential": _zero_potential(charges, config.batch_mode),
        "iterations": 0,
    }
