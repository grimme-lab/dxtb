# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Parametrization: Short-range corrections
========================================

Definitions of short-range corrections that are part of an xTB
parametrization.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

__all__ = ["PShortRange", "PShortRangeBond"]


class PShortRangeBond(BaseModel):
    """Parameters of the GFN0 short-range bond (SRB) correction."""

    model_config = ConfigDict(extra="forbid")

    shift: float
    """Additional offset for approximate reference bond lengths."""

    prefactor: float
    """Energy prefactor in Hartree."""

    steepness: float
    """Steepness of the Pauling-EN-dependent Gaussian."""

    enscale: float
    """Scaling factor applied to the Pauling electronegativity difference."""

    enpoly: List[float]
    """
    Coefficients of the fitted-electronegativity polynomial that scales the
    SRB reference distance. Entry ``n`` multiplies
    ``abs(srb_en_i - srb_en_j) ** (n + 1)``; GFN0 therefore stores its linear
    and quadratic coefficients in that order.
    """

    cn: str
    """Name of the tad-mctc coordination-number counting function."""

    cn_cutoff: float
    """Cutoff for the SRB coordination number in Bohr."""

    cn_max: float
    """Smooth upper bound applied to the SRB coordination number."""

    cn_kcn: float
    """Steepness of the SRB coordination-number counting function."""

    pair_cutoff2: float
    """Squared cutoff for selecting SRB pairs in Bohr squared."""


class PShortRange(BaseModel):
    """Available short-range corrections."""

    model_config = ConfigDict(extra="forbid")

    srb: Optional[PShortRangeBond] = None
    """Short-range bond correction."""
