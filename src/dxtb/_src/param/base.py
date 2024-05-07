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
Parametrization: Base
=====================

Definition of the full parametrization data for the extended tight-binding
methods.

The dataclass can represent a complete parametrization file produced by the
`tblite`_ library, however it only stores the raw data rather than the full
representation.

The parametrization of a calculator with the model data must account for missing
transformations, like extracting the principal quantum numbers from the shells.
The respective checks are therefore deferred to the instantiation of the
calculator, while a deserialized model in `tblite`_ is already verified at this
stage.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel

from .charge import Charge
from .dispersion import Dispersion
from .element import Element
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .multipole import Multipole
from .repulsion import Repulsion
from .thirdorder import ThirdOrder

__all__ = ["Param"]


class Param(BaseModel):
    """
    Complete self-contained representation of an extended tight-binding model.
    """

    meta: Optional[Meta] = None
    """Descriptive data on the model."""

    element: Dict[str, Element]
    """Element specific parameter records."""

    hamiltonian: Optional[Hamiltonian] = None
    """Definition of the Hamiltonian, always required."""

    dispersion: Optional[Dispersion] = None
    """Definition of the dispersion correction."""

    repulsion: Optional[Repulsion] = None
    """Definition of the repulsion contribution."""

    charge: Optional[Charge] = None
    """Definition of the isotropic second-order charge interactions."""

    multipole: Optional[Multipole] = None
    """Definition of the anisotropic second-order multipolar interactions."""

    halogen: Optional[Halogen] = None
    """Definition of the halogen bonding correction."""

    thirdorder: Optional[ThirdOrder] = None
    """Definition of the isotropic third-order charge interactions."""
