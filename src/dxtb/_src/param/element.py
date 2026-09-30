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

from typing import List, Optional

from pydantic import BaseModel
from tad_mctc.data import pse

from dxtb._src.typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from torch import nn

__all__ = ["Element", "get_elem_angular", "shell_angular"]

LABEL2ANGULAR = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
"""Angular momentum for each shell type label."""


def shell_angular(label: Any) -> int:
    """
    Obtain the angular momentum of a shell.

    Parameters
    ----------
    label : Any
        Shell label with principal quantum number and shell type (e.g.
        ``"2p"``). Anything indexable like a string works, such as the
        wrapped labels of a :class:`~dxtb.ParamModule`.

    Returns
    -------
    int
        Angular momentum of the shell.

    Raises
    ------
    ValueError
        If the shell type is unknown.
    """
    shell_type = label[-1]
    if shell_type not in LABEL2ANGULAR:
        raise ValueError(f"Unknown shell type '{shell_type}'.")

    return LABEL2ANGULAR[shell_type]


def get_elem_angular(
    element: Mapping[str, Any] | nn.ModuleDict,
) -> dict[int, list[int]]:
    """
    Obtain angular momenta of the shells of all elements.

    Parameters
    ----------
    element : Mapping[str, Any] | nn.ModuleDict
        Element parameters by element symbol. Each entry needs ``shells``,
        as in :attr:`dxtb.Param.element` or the ``element`` tree of a
        :class:`~dxtb.ParamModule`.

    Returns
    -------
    dict[int, list[int]]
        Angular momenta of all elements, keyed by atomic number.

    Raises
    ------
    ValueError
        If a shell type is unknown.
    """
    result: dict[int, list[int]] = {}

    for sym, elem in element.items():
        # In a `ParamModule`, `shells` is a `ModuleList` of wrapped labels,
        # which the type checker cannot tell is iterable.
        shells: Any = elem.shells
        result[pse.S2Z[sym]] = [shell_angular(label) for label in shells]

    return result


class Element(BaseModel):
    """
    Representation of the parameters for a species.
    """

    shells: List[str]
    """Included shells with principal quantum number and angular momentum."""

    levels: List[float]
    """Atomic level energies for each shell"""

    slater: List[float]
    """Slater exponents of the STO-NG functions for each shell"""

    ngauss: List[int]
    """
    Number of primitive Gaussian functions used in the STO-NG expansion for
    each shell.
    """

    ############################################################################

    refocc: List[float]
    """Reference occupation for each shell"""

    shpoly: List[float]
    """Polynomial enhancement for Hamiltonian elements"""

    kcn: List[float]
    """CN dependent shift of the self energy for each shell"""

    kq: Optional[List[float]] = None
    """Linear charge-dependent shift of the self energy for each shell."""

    kqat: Optional[float] = None
    """Quadratic atom-resolved charge shift of the self energy."""

    h0rad: Optional[float] = None
    """Atomic radius used by the GFN0 H0 distance polynomial."""

    ############################################################################

    gam: Optional[float] = None
    """Chemical hardness / Hubbard parameter."""

    lgam: Optional[List[float]] = None
    """Relative chemical hardness for each shell."""

    gam3: float = 0.0
    """Atomic Hubbard derivative."""

    ############################################################################

    zeff: float
    """Effective nuclear charge used in repulsion."""

    arep: float
    """Repulsion exponent."""

    ############################################################################

    xbond: float = 0.0
    """Halogen bonding strength."""

    en: float
    """Electronegativity."""

    eeq_chi: Optional[float] = None
    """Electronegativity of the main GFN0 EEQ model."""

    eeq_eta: Optional[float] = None
    """Chemical hardness of the main GFN0 EEQ model."""

    eeq_kcn: Optional[float] = None
    """Coordination-number dependence of the main GFN0 EEQ model."""

    eeq_rad: Optional[float] = None
    """Charge width of the main GFN0 EEQ model."""

    srb_r0: Optional[float] = None
    """Reference radius of the GFN0 short-range bond correction."""

    srb_cnfak: Optional[float] = None
    """Coordination-number radius shift of the GFN0 SRB correction."""

    srb_en: Optional[float] = None
    """Fitted electronegativity used by the GFN0 SRB correction."""

    ############################################################################

    dkernel: float = 0.0
    """Dipolar exchange-correlation kernel."""

    qkernel: float = 0.0
    """Quadrupolar exchange-correlation kernel."""

    mprad: float = 0.0
    """Offset radius for the damping in the AES energy."""

    mpvcn: float = 0.0
    """Shift value in the damping in the AES energy. Only used if mprad != 0."""
