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
Parametrization: Electrostatics (multipole)
===========================================

Definition of the anisotropic second-order multipolar interactions.
Currently, only GFN2-xTB's damped multipole version is supported.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["MultipoleDamped", "Multipole"]


class MultipoleDamped(BaseModel):
    """
    Representation of the anisotropic second-order multipolar interactions
    for a parametrization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dmp3: float
    """Damping function for inverse quadratic contributions."""

    dmp5: float
    """Damping function for inverse cubic contributions"""

    kexp: float
    """Exponent for the generation of the multipolar damping radii."""

    shift: float
    """Shift for the generation of the multipolar damping radii."""

    rmax: float
    """Maximum radius for the multipolar damping radii (Eq. 29)."""


class Multipole(BaseModel):
    """
    Possible parametrizations for multipole electrostatics.
    """

    damped: MultipoleDamped
    """Damped second-order multipolar electrostatics (GFN2)."""
