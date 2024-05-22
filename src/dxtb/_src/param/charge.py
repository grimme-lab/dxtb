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
Parametrization: Electrostatics (2nd order)
===========================================

Definition of the isotropic second-order charge interactions.
"""

from __future__ import annotations

from pydantic import BaseModel

from dxtb._src.constants.xtb import DEFAULT_ES2_GEXP

__all__ = ["ChargeEffective", "Charge"]


class ChargeEffective(BaseModel):
    """
    Representation of the isotropic second-order charge interactions for a
    parametrization.
    """

    gexp: float = DEFAULT_ES2_GEXP
    """Exponent of Coulomb kernel. """

    average: str = "harmonic"
    """Averaging function for Hubbard parameter."""


class Charge(BaseModel):
    """
    Possible charge parametrizations. Currently only the interaction kernel
    for the Klopman-Ohno electrostatics (effective) is supported.
    """

    effective: ChargeEffective
    """Klopman-Ohno electrostatics."""
