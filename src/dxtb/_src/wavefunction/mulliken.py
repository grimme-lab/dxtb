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
Wavefunction: Mulliken
======================

Wavefunction analysis via Mulliken populations.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.typing import Tensor

__all__ = [
    "get_orbital_populations",
    "get_shell_populations",
    "get_atomic_populations",
    "get_mulliken_shell_charges",
    "get_mulliken_atomic_charges",
]


def get_orbital_populations(
    overlap: Tensor,
    density: Tensor,
) -> Tensor:
    """
    Compute orbital-resolved populations using Mulliken population analysis.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.

    Returns
    -------
    Tensor
        Orbital populations.
    """

    return torch.diagonal(density @ overlap, dim1=-2, dim2=-1)


def get_shell_populations(
    overlap: Tensor,
    density: Tensor,
    indexhelper: IndexHelper,
) -> Tensor:
    """
    Compute shell-resolved populations using Mulliken population analysis.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    indexhelper : IndexHelper
        Index mapping for the basis set.

    Returns
    -------
    Tensor
        Shell populations.
    """

    return indexhelper.reduce_orbital_to_shell(
        get_orbital_populations(overlap, density)
    )


def get_atomic_populations(
    overlap: Tensor,
    density: Tensor,
    indexhelper: IndexHelper,
) -> Tensor:
    """
    Compute atom-resolved populations.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    indexhelper : IndexHelper
        Index mapping for the basis set.

    Returns
    -------
    Tensor
        Atom populations.
    """

    return indexhelper.reduce_shell_to_atom(
        get_shell_populations(overlap, density, indexhelper)
    )


def get_mulliken_shell_charges(
    overlap: Tensor,
    density: Tensor,
    indexhelper: IndexHelper,
    n0: Tensor,
) -> Tensor:
    """
    Compute shell-resolved Mulliken partial charges using Mulliken population
    analysis.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    indexhelper : IndexHelper
        Index mapping for the basis set.
    n0 : Tensor
        Shell-resolved reference occupancy numbers.

    Returns
    -------
    Tensor
        Shell-resolved Mulliken partial charges.
    """

    return n0 - get_shell_populations(overlap, density, indexhelper)


def get_mulliken_atomic_charges(
    overlap: Tensor,
    density: Tensor,
    indexhelper: IndexHelper,
    n0: Tensor,
) -> Tensor:
    """
    Compute atom-resolved Mulliken partial charges.

    Parameters
    ----------
    overlap : Tensor
        Overlap matrix.
    density : Tensor
        Density matrix.
    indexhelper : IndexHelper
        Index mapping for the basis set.
    n0 : Tensor
        Atom-resolved reference occupancy numbers.

    Returns
    -------
    Tensor
        Atom-resolved Mulliken partial charges.
    """

    return n0 - get_atomic_populations(overlap, density, indexhelper)
