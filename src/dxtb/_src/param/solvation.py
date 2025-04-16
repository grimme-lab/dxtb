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
Parametrization: Electrostatics (3rd order)
===========================================

Definition of the isotropic third-order onsite correction.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

__all__ = ["ALPB", "Solvation"]


class ALPB(BaseModel):
    """
    Representation of the analytical linearized Poisson-Boltzmann solvation
    model (10.1021/acs.jctc.1c00471).
    """

    alpb: bool
    """Use analytical linearized Poisson-Boltzmann model."""

    kernel: str
    """
    Born interaction kernels. Either classical Still kernel or P16 kernel
    by Lange (JCTC 2012, 8, 1999-2011).
    """

    born_scale: float
    """Scaling factor for Born radii."""

    born_offset: float
    """Offset parameter for Born radii integration."""


class Solvation(BaseModel):
    """
    Representation of the solvation models.
    """

    alpb: Optional[ALPB] = None
    """
    Whether the third order contribution is shell-dependent or only atomwise.
    """
