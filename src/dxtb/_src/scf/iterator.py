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
Self-consistent field iteration
===============================

Provides implementation of self consistent field iterations for the xTB
Hamiltonian. The iterations are not like in ab initio calculations expressed in
the density matrix and the derivative of the energy w.r.t. to the density
matrix, i.e. the Hamiltonian, but the Mulliken populations (or partial charges)
of the respective orbitals as well as the derivative of the energy w.r.t. to
those populations, i.e. the potential vector.
"""

from __future__ import annotations

import torch
from tad_mctc import storch

from dxtb import IndexHelper
from dxtb._src.components.interactions import InteractionList, InteractionListCache
from dxtb._src.constants import labels
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Any, Tensor
from dxtb._src.wavefunction import filling
from dxtb.config import ConfigSCF

from .base import SCFResult
from .guess import get_guess

__all__ = ["solve"]


def solve(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    spin: Tensor | None,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    refocc: Tensor,
    *args: Any,
    **kwargs: Any,
) -> SCFResult:
    """
    Obtain self-consistent solution for a given Hamiltonian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    chrg : Tensor
        Total charge.
    interactions : InteractionList
        Collection of `Interation` objects.
    ihelp : IndexHelper
        Index helper object.
    config : ConfigSCF
        Configuration for the SCF calculation.
    integrals : Integrals
        Container for all integrals.
    args : Tuple
        Positional arguments to pass to the engine.
    kwargs : dict
        Keyword arguments to pass to the engine.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """
    n0, occupation = get_refocc(refocc, chrg, spin, ihelp)
    charges = get_guess(numbers, positions, chrg, ihelp, config.guess)

    if config.scf_mode == labels.SCF_MODE_IMPLICIT:
        # pylint: disable=import-outside-toplevel
        from .pure import scf_wrapper

        return scf_wrapper(
            interactions,
            occupation,
            n0,
            charges,
            numbers=numbers,
            ihelp=ihelp,
            cache=cache,
            integrals=integrals,
            config=config,
            *args,
            **kwargs,
        )

    if config.scf_mode == labels.SCF_MODE_IMPLICIT_NON_PURE:
        # pylint: disable=import-outside-toplevel
        from .implicit import SelfConsistentFieldImplicit as SCF
    elif config.scf_mode == labels.SCF_MODE_FULL:
        # pylint: disable=import-outside-toplevel
        from .unrolling import SelfConsistentFieldFull as SCF
    elif config.scf_mode == labels.SCF_MODE_EXPERIMENTAL:
        # pylint: disable=import-outside-toplevel
        from .unrolling import SelfConsistentFieldSingleShot as SCF
    else:
        name = labels.SCF_MODE_MAP[config.scf_mode]
        raise ValueError(f"Unknown SCF mode '{name}' (input name can vary).")

    return SCF(
        interactions,
        occupation,
        n0,
        *args,
        numbers=numbers,
        ihelp=ihelp,
        cache=cache,
        integrals=integrals,
        config=config,
        **kwargs,
    )(charges)


def get_refocc(
    refs: Tensor, chrg: Tensor, spin: Tensor | None, ihelp: IndexHelper
) -> tuple[Tensor, Tensor]:
    """
    Obtain reference occupation and total number of electrons.

    Parameters
    ----------
    refs : Tensor
        Occupation from parametrization.
    chrg : Tensor
        Total charge.
    spin : Tensor | None
        Number of unpaired electrons.
    ihelp : IndexHelper
        Helper for indexing.

    Returns
    -------
    tuple[Tensor, Tensor]
        Reference occupation and occupation.
    """

    refocc = ihelp.spread_ushell_to_orbital(refs)
    orb_per_shell = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)

    n0 = torch.where(
        orb_per_shell != 0,
        storch.divide(refocc, orb_per_shell),
        torch.tensor(0, device=refs.device, dtype=refs.dtype),
    )

    # Obtain the reference occupation and total number of electrons
    nel = torch.sum(n0, -1) - torch.sum(chrg, -1)

    # get alpha and beta electrons and occupation
    nab = filling.get_alpha_beta_occupation(nel, spin)
    occupation = filling.get_aufbau_occupation(
        torch.tensor(ihelp.nao, device=refs.device, dtype=torch.int64),
        nab,
    )

    return n0, occupation
