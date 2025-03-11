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

from typing import List, Union

from pydantic import BaseModel

from .tensor import TensorPydantic

__all__ = ["Element"]


class Element(BaseModel):
    """
    Representation of the parameters for a species.
    """

    shells: List[str]
    """Included shells with principal quantum number and angular momentum."""

    levels: Union[Union[List[float], TensorPydantic], TensorPydantic]
    """Atomic level energies for each shell"""

    slater: Union[List[float], TensorPydantic]
    """Slater exponents of the STO-NG functions for each shell"""

    ngauss: Union[List[int], TensorPydantic]
    """
    Number of primitive Gaussian functions used in the STO-NG expansion for
    each shell.
    """

    ############################################################################

    refocc: Union[List[float], TensorPydantic]
    """Reference occupation for each shell"""

    shpoly: Union[List[float], TensorPydantic]
    """Polynomial enhancement for Hamiltonian elements"""

    kcn: Union[List[float], TensorPydantic]
    """CN dependent shift of the self energy for each shell"""

    ############################################################################

    gam: Union[float, TensorPydantic]
    """Chemical hardness / Hubbard parameter."""

    lgam: Union[List[float], TensorPydantic]
    """Relative chemical hardness for each shell."""

    gam3: Union[float, TensorPydantic] = 0.0
    """Atomic Hubbard derivative."""

    ############################################################################

    zeff: Union[float, TensorPydantic]
    """Effective nuclear charge used in repulsion."""

    arep: Union[float, TensorPydantic]
    """Repulsion exponent."""

    ############################################################################

    xbond: Union[float, TensorPydantic] = 0.0
    """Halogen bonding strength."""

    en: Union[float, TensorPydantic]
    """Electronnegativity."""

    ############################################################################

    dkernel: Union[float, TensorPydantic] = 0.0
    """Dipolar exchange-correlation kernel."""

    qkernel: Union[float, TensorPydantic] = 0.0
    """Quadrupolar exchange-correlation kernel."""

    mprad: Union[float, TensorPydantic] = 0.0
    """Offset radius for the damping in the AES energy."""

    mpvcn: Union[float, TensorPydantic] = 0.0
    """Shift value in the damping in the AES energy. Only used if mprad != 0."""
