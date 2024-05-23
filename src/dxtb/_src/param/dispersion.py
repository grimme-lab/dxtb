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

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from dxtb._src.typing import Tensor

from ..constants import xtb

__all__ = ["D3Model", "D4Model", "Dispersion"]


class D3Model(BaseModel):
    """
    Representation of the DFT-D3(BJ) contribution for a parametrization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    s6: Union[float, Tensor] = xtb.DEFAULT_DISP_S6
    """Scaling factor for multipolar (dipole-dipole contribution) terms."""

    s8: Union[float, Tensor] = xtb.DEFAULT_DISP_S8
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms."""

    s9: Union[float, Tensor] = xtb.DEFAULT_DISP_S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""

    a1: Union[float, Tensor] = xtb.DEFAULT_DISP_A1
    """Becke-Johnson damping parameter."""

    a2: Union[float, Tensor] = xtb.DEFAULT_DISP_A2
    """Becke-Johnson damping parameter."""


class D4Model(BaseModel):
    """
    Representation of the DFT-D4 contribution for a parametrization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sc: bool = False
    """Whether the dispersion correctio is used self-consistently or not."""

    s6: Union[float, Tensor] = xtb.DEFAULT_DISP_S6
    """Scaling factor for multipolar (dipole-dipole contribution) terms"""

    s8: Union[float, Tensor] = xtb.DEFAULT_DISP_S8
    """Scaling factor for multipolar (dipole-quadrupole contribution) terms"""

    s9: Union[float, Tensor] = xtb.DEFAULT_DISP_S9
    """Scaling factor for the many-body dispersion term (ATM/RPA-like)."""

    s10: Union[float, Tensor] = xtb.DEFAULT_DISP_S10
    """Scaling factor for quadrupole-quadrupole term."""

    s10: Union[float, Tensor] = xtb.DEFAULT_DISP_S10
    """Scaling factor for quadrupole-quadrupole contribution."""

    alp: Union[float, Tensor] = xtb.DEFAULT_DISP_ALP
    """Exponent of zero damping function in the ATM term."""

    a1: Union[float, Tensor] = xtb.DEFAULT_DISP_A1
    """Becke-Johnson damping parameter."""

    a2: Union[float, Tensor] = xtb.DEFAULT_DISP_A2
    """Becke-Johnson damping parameter."""


class Dispersion(BaseModel):
    """
    Possible dispersion parametrizations. Currently, the DFT-D3(BJ) and DFT-D4
    methods are supported.
    """

    d3: Optional[D3Model] = None
    """D3 model for the dispersion."""

    d4: Optional[D4Model] = None
    """D4 model for the dispersion."""
