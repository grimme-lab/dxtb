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
Parametrization: Dispersion
===========================

Definitions of dispersion contributions. Contains the :class:`D3Model` and
:class:`D4Model` representing the DFT-D3(BJ) and DFT-D4 dispersion corrections,
respectively.
For details on there implementation, see the `tad-dftd3`_ and `tad-dftd4`_
libraries.

.. _tad-dftd3: https://github.com/dftd3/tad-dftd3

.. _tad-dftd4: https://github.com/dftd4/tad-dftd4
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from tad_dftd3 import defaults as d3_defaults
from tad_dftd4 import defaults as d4_defaults

__all__ = ["D3Model", "D4Model", "Dispersion"]


class D3Model(BaseModel):
    """
    Representation of the DFT-D3(BJ) contribution for a parametrization.
    """

    s6: float = d3_defaults.S6
    """Scaling factor for multipolar (dipole-dipole contribution) terms."""

    s8: float = d3_defaults.S8
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms."""

    a1: float = d3_defaults.A1
    """Becke-Johnson damping parameter."""

    a2: float = d3_defaults.A2
    """Becke-Johnson damping parameter."""

    s9: float = d3_defaults.S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""


class D4Model(BaseModel):
    """
    Representation of the DFT-D4 contribution for a parametrization.
    """

    sc: bool = False
    """Whether the dispersion correctio is used self-consistently or not."""

    s6: float = d4_defaults.S6
    """Scaling factor for multipolar (dipole-dipole contribution) terms"""

    s8: float = d4_defaults.S8
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms"""

    a1: float = d4_defaults.A1
    """Becke-Johnson damping parameter."""

    a2: float = d4_defaults.A2
    """Becke-Johnson damping parameter."""

    s9: float = d4_defaults.S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""

    s10: float = d4_defaults.S10
    """Scaling factor for quadrupole-quadrupole term."""

    alp: float = d4_defaults.ALP
    """Exponent of zero damping function in the ATM term."""


class Dispersion(BaseModel):
    """
    Possible dispersion parametrizations. Currently, the DFT-D3(BJ) and DFT-D4
    methods are supported.
    """

    d3: Optional[D3Model] = None
    """D3 model for the dispersion."""

    d4: Optional[D4Model] = None
    """D4 model for the dispersion."""
