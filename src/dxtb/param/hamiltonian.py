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
Parametrization: Hamiltonian
============================

Definition of the global core Hamiltonian parameters.

The core Hamiltonian is rescaling the shell-blocks of the overlap integrals
formed over the basis set by the average of the atomic self-energies and an
additional distance dependent function formed from the element parametrization.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel

__all__ = ["Hamiltonian", "XTBHamiltonian"]


class XTBHamiltonian(BaseModel):
    """
    Global parameters for the formation of the core Hamiltonian from the
    overlap integrals. Contains the required atomic and shell dependent scaling
    parameters to obtain the off-site scaling functions independent of the
    self-energy and the distance polynomial.
    """

    shell: Dict[str, float]
    """Shell-pair dependent scaling factor for off-site blocks"""

    kpair: Dict[str, float] = {}
    """Atom-pair dependent scaling factor for off-site valence blocks"""

    enscale: float
    """Electronegativity scaling factor for off-site valence blocks"""

    wexp: float
    """Exponent of the orbital exponent dependent off-site scaling factor"""

    cn: Optional[str] = None
    """Local environment descriptor for shifting the atomic self-energies"""

    kpol: float = 2.0
    """Scaling factor for polarization functions"""


class Hamiltonian(BaseModel):
    """
    Possible Hamiltonian parametrizations.
    Currently only the xTB Hamiltonian is supported.
    """

    xtb: XTBHamiltonian
    """Data for the xTB Hamiltonian"""
