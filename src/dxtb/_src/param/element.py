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
Parametrization: Element
========================

Element parametrization record containing the adjustable parameters for each
species.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

__all__ = ["Element"]


class Element(BaseModel):
    """
    Representation of the parameters for a species.
    """

    zeff: float
    """Effective nuclear charge used in repulsion."""

    arep: float
    """Repulsion exponent."""

    ############################################################################

    en: float
    """Electronnegativity."""

    shells: List[str]
    """Included shells with principal quantum number and angular momentum."""

    ngauss: List[int]
    """
    Number of primitive Gaussian functions used in the STO-NG expansion for
    each shell.
    """

    ############################################################################

    levels: List[float]
    """Atomic level energies for each shell"""

    slater: List[float]
    """Slater exponents of the STO-NG functions for each shell"""

    refocc: List[float]
    """Reference occupation for each shell"""

    kcn: List[float]
    """CN dependent shift of the self energy for each shell"""

    shpoly: List[float]
    """Polynomial enhancement for Hamiltonian elements"""

    ############################################################################

    gam: float
    """Chemical hardness / Hubbard parameter."""

    lgam: List[float]
    """Relative chemical hardness for each shell."""

    gam3: float = 0.0
    """Atomic Hubbard derivative."""

    ############################################################################

    dkernel: float = 0.0
    """Dipolar exchange-correlation kernel."""

    qkernel: float = 0.0
    """Quadrupolar exchange-correlation kernel."""

    mprad: float = 0.0
    """Offset radius for the damping in the AES energy."""

    mpvcn: float = 0.0
    """Shift value in the damping in the AES energy. Only used if mprad != 0."""

    ############################################################################

    xbond: float = 0.0
    """Halogen bonding strength."""
