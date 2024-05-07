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
Parametrization: Halogen
========================

Definitions for halogen binding corrections.
Currently, only GFN1-xTB's classical halogen bond correction is defined.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["ClassicalHalogen", "Halogen"]


class ClassicalHalogen(BaseModel):
    """
    Representation of the classical geometry dependent halogen-bond (XB)
    correction for a parametrization.
    """

    damping: float
    """
    Damping factor of attractive contribution in Lennard-Jones-like potential.
    """

    rscale: float
    """Global scaling factor for covalent radii of AX bond."""


class Halogen(BaseModel):
    """
    Possible halogen correction parametrizations.
    """

    classical: ClassicalHalogen
    """Classical halogen-bond correction used in GFN1-xTB."""
