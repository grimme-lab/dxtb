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
`xtb` parameters
================

This module contains `xtb` parameters of the model that are not contained
in the parametrization file. Furthermore, `xtb`'s default values are stored here.
"""


# Coordination number

KCN_EEQ = 7.5
"""Steepness of counting function in EEQ model (7.5)."""

KCN = 16.0
"""GFN1: Steepness of counting function."""

KA = 10.0
"""GFN2: Steepness of first counting function."""

KB = 20.0
"""GFN2: Steepness of second counting function."""

R_SHIFT = 2.0
"""GFN2: Offset of the second counting function."""

NCOORD_DEFAULT_CUTOFF = 25.0
"""Default cutoff used for determining coordination number (25.0)."""


# Electrostatics

DEFAULT_ES2_GEXP: float = 2.0
"""Default exponent of the second-order Coulomb interaction (2.0)."""


# Dispersion

DEFAULT_DISP_A1 = 0.4
"""Scaling for the C8 / C6 ratio in the critical radius (0.4)."""

DEFAULT_DISP_A2 = 5.0
"""Offset parameter for the critical radius (5.0)."""

DEFAULT_DISP_S6 = 1.0
"""Default scaling of dipole-dipole term (1.0 to retain correct limit)."""

DEFAULT_DISP_S8 = 1.0
"""Default scaling of dipole-quadrupole term (1.0)."""

DEFAULT_DISP_S9 = 1.0
"""Default scaling of three-body term (1.0)."""

DEFAULT_DISP_S10 = 0.0
"""Default scaling of quadrupole-quadrupole term (0.0)."""

DEFAULT_DISP_RS9 = 4.0 / 3.0
"""Scaling for van-der-Waals radii in damping function (4.0/3.0)."""

DEFAULT_DISP_ALP = 16.0
"""Exponent of zero damping function (16.0)."""

# Classical contributions

DEFAULT_XB_CUTOFF: float = 20.0
"""Default real space cutoff for halogen bonding interactions (20.0)."""

DEFAULT_REPULSION_CUTOFF: float = 25.0
"""Default real space cutoff for repulsion interactions."""
